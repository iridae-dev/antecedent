//! Ordinary integration reachability of the actual closed Bayesian refusal engine.
use antecedent_core::ExecutionContext;
use antecedent_learn::nested_markov_bayesian::{BayesianRefusal, Options, Prior, fit};
#[test]
fn nested_bayesian_refuses_malformed_counts_priors_requests_budget_and_cancellation() {
    let ctx = ExecutionContext::for_tests(0);
    let prior = Prior::default();
    let options = Options::default();
    for invalid in [[0; 16], [1_000_000_001; 16]] {
        assert!(matches!(fit(&invalid, &prior, &options, &ctx), Err(BayesianRefusal::Invalid(_))));
    }
    for shape in [0.5, f64::NAN, f64::INFINITY, 1_000_001.0] {
        let mut bad = prior.clone();
        bad.alpha[8] = shape;
        assert!(matches!(fit(&[1; 16], &bad, &options, &ctx), Err(BayesianRefusal::Invalid(_))));
    }
    for request in [
        Options { credible_mass: 0.9, ..options.clone() },
        Options { chains: 3, ..options.clone() },
        Options { draws: 255, ..options.clone() },
        Options { warmup: 0, ..options.clone() },
    ] {
        assert!(matches!(fit(&[1; 16], &prior, &request, &ctx), Err(BayesianRefusal::Invalid(_))));
    }
    assert!(matches!(
        fit(&[1; 16], &prior, &Options { max_proposals: 1, ..options.clone() }, &ctx),
        Err(BayesianRefusal::Budget(_))
    ));
    let mut small = ExecutionContext::for_tests(0);
    small.memory.hard_limit_bytes = Some(0);
    assert!(matches!(fit(&[1; 16], &prior, &options, &small), Err(BayesianRefusal::Budget(_))));
    assert!(matches!(
        fit(&[1; 16], &prior, &Options { warmup: 128, draws: 256, ..options.clone() }, &ctx),
        Err(BayesianRefusal::Nonconvergence)
    ));
    ctx.cancellation.cancel();
    assert!(matches!(fit(&[1; 16], &prior, &options, &ctx), Err(BayesianRefusal::Budget(_))));
}
