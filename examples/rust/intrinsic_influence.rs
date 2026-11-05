//! Intrinsic causal influence: which noise term drives a target's variance.
//!
//! Data from `x -> y -> z` with `y = 2x + e_y` (unit noise) and `z = 1.5y + e_z` (noise scale
//! 0.5). A study with an `Attribute` query on the supplied DAG fits the mechanisms and splits
//! the variance of `z`. The total coefficients of the three noise terms on `z` are 3, 1.5 and 1,
//! so their Shapley shares of `Var(z)` are 9, 2.25 and 0.25. The estimate is nested Monte Carlo
//! through batch noise replay, and every interval names the one uncertainty component it covers.
//!
//! Run: `cargo run -p antecedent --example intrinsic_influence`
//! Source: `examples/rust/intrinsic_influence.rs`

use antecedent::gcm::{ComponentStatus, UncertaintyComponent};
use antecedent::{AttributionOutcome, CompiledExecution, Study};
use antecedent_core::{
    AttributeQuery, AttributeTarget, CausalQuery, CausalRng, ExecutionContext, InfluenceAllocation,
    IntrinsicInfluenceTarget, NoiseDeclaration, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

const TARGET: VariableId = VariableId::from_raw(2);
/// Closed-form contributions of the noise terms of `x`, `y` and `z` to `Var(z)`.
const TRUTH: [f64; 3] = [9.0, 2.25, 0.25];

/// Rows of data the mechanisms are fitted to.
const ROWS: usize = 4000;

/// The data of the structural model above, from deterministic standard normal noise.
fn sample_data() -> Result<TabularData, Box<dyn std::error::Error>> {
    let mut rng = CausalRng::from_seed(21);
    let mut gauss = || {
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    let (mut x, mut y, mut z) = (vec![0.0; ROWS], vec![0.0; ROWS], vec![0.0; ROWS]);
    for i in 0..ROWS {
        x[i] = gauss();
        y[i] = 2.0 * x[i] + gauss();
        z[i] = 1.5 * y[i] + 0.5 * gauss();
    }
    Ok(TabularData::from_f64_columns([
        ("x", x.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])?)
}

/// The study of the variance of `z`, with the noise declaration the caller makes.
fn influence_study(
    data: &TabularData,
    graph: &Dag,
    noise: NoiseDeclaration,
    (outer, replicates): (usize, usize),
) -> Result<Study, Box<dyn std::error::Error>> {
    let target = IntrinsicInfluenceTarget {
        target: TARGET,
        allocation: InfluenceAllocation::Exact,
        outer_draws: outer,
        inner_draws: 20,
        replicates,
        fit_bootstrap_replicates: 0,
        level: 0.95,
        max_sources: 8,
        seed: 5,
        noise,
    };
    let query =
        CausalQuery::Attribute(AttributeQuery::new(AttributeTarget::IntrinsicInfluence(target)));
    Ok(Study::tabular(data.clone()).graph(graph.clone()).query(query).build()?)
}

/// Runs the example end to end; `main` calls it and the example test suite runs it.
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let data = sample_data()?;
    let mut graph = Dag::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1))?;
    graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2))?;
    let ctx = ExecutionContext::for_tests(11);
    let (outer, replicates) = (400, 4);

    // The caller must say the noise terms are independent; a compiled DAG cannot certify it.
    let refused = influence_study(&data, &graph, NoiseDeclaration::Undeclared, (outer, replicates))
        .and_then(|study| Ok(CompiledExecution::run(&study, &ctx)?))
        .expect_err("undeclared noise independence is refused");
    assert!(refused.to_string().contains("noise_independence_undeclared"), "{refused}");

    let study = influence_study(&data, &graph, NoiseDeclaration::Independent, (outer, replicates))?;
    let execution = CompiledExecution::run(&study, &ctx)?;
    let Some(payload) = execution.result().attribution() else {
        panic!("an attribution query returns an attribution");
    };
    let AttributionOutcome::IntrinsicInfluence(result) = &payload.outcome else {
        panic!("an intrinsic-influence target returns its allocation");
    };
    assert_eq!(result.sources.len(), 3, "the ancestors and the target itself are the players");
    for (source, truth) in result.sources.iter().zip(TRUTH) {
        // Six times the larger of the simulation error implied by the design and the reported one.
        let derived = (2.0 / (replicates as f64 * (outer as f64 - 1.0))).sqrt() * truth;
        // The mechanisms are fitted to a finite sample, which moves each share by a few percent.
        let tolerance = 6.0 * derived.max(source.quantified_stderr().stderr) + 0.15 * truth;
        assert!(
            (source.contribution - truth).abs() <= tolerance,
            "{:?}: {} vs {truth}",
            source.source,
            source.contribution
        );
        // Three sources enumerate every coalition, so no permutation is sampled; the
        // simulation error is measured; mechanism-fit uncertainty was not asked for, which is not zero.
        assert!(matches!(
            source.status(UncertaintyComponent::PermutationSampling),
            Some(ComponentStatus::NotApplicable { .. })
        ));
        assert!(matches!(
            source.status(UncertaintyComponent::Simulation),
            Some(ComponentStatus::Quantified { .. })
        ));
        assert!(!matches!(
            source.status(UncertaintyComponent::MechanismFit),
            Some(ComponentStatus::Quantified { .. })
        ));
    }

    // The shares add up to Var(z) up to the estimation error reported as the residual.
    let total: f64 = TRUTH.iter().sum();
    assert!(
        (result.total_variance - total).abs()
            <= 6.0 * result.total_variance_stderr.max(0.05 * total)
    );
    assert!(
        (result.total_variance - result.contribution_sum() - result.allocation_residual).abs()
            < 1e-9
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run()
}
