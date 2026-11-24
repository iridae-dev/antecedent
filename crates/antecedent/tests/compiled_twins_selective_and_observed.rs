//! Twin equality of the selective-discovery adapters and of the partially observed signed-uniform
//! marginals with the compiled structure-discovery route, and an independent check of the
//! compiled marginals.
//!
//! * A selective search compiles as the `DiscoverStatic` or `DiscoverTemporal` route of its
//!   algorithm and runs the adapter: its outcome, structure, run report and exported state equal
//!   the adapter's own run, fresh and resumed from a portable state.
//! * Exact signed-uniform marginals with a draw limit integrate the missing cells of the last
//!   column with the same kernel: every probability, the log normalizer and the Monte Carlo
//!   summary equal the direct call by `to_bits()` under the run's master seed.
//! * Independent check: on three rows with one missing outcome, the compiled marginals agree with
//!   an independent integration of the missing value and the coefficient in the joint density.
//!
//! Seeds: the adapter refuses a context whose master seed is not the contract's, so the direct
//! run and the study share one context; the marginals are seeded by that context's master seed.
// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::float_cmp,
    clippy::too_many_lines,
    reason = "integration test with deterministic fixtures and bitwise comparisons"
)]

use antecedent::analysis::selective_discovery::{
    CiTestSpec, StaticAlgorithm, StaticDiscoveryContract, StaticSelectiveDiscovery,
    TemporalAlgorithm, TemporalDiscoveryContract, TemporalSelectiveDiscovery,
};
use antecedent::discovery::{BayesianDiscoverParams, GraphPrior, StaticDiscoverParams};
use antecedent::discovery_defaults::resolve_ci;
use antecedent::{CompiledExecution, DiscoveredStructure, DiscoverySpec, Study};
use antecedent_core::{CausalRng, ExecutionContext, VariableId};
use antecedent_data::{SamplingRegularity, TabularData, TimeIndex, TimeSeriesData};
use antecedent_discovery::BackgroundKnowledge;
use antecedent_discovery::observed_marginals::signed_uniform_observed_marginals;
use antecedent_state::GraphScoreFamily;

const SEED: u64 = 0x5E1E_C7ED;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(SEED)
}

fn vars(n: u32) -> Vec<VariableId> {
    (0..n).map(VariableId::from_raw).collect()
}

fn on_large_stack(name: &str, body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name(name.into())
        .stack_size(32 * 1024 * 1024)
        .spawn(body)
        .unwrap()
        .join()
        .unwrap();
}

fn normal(rng: &mut CausalRng) -> f64 {
    let u1 = rng.next_f64().max(1e-12);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// `a -> c <- b`, `c -> d`.
fn collider_table(n: usize, seed: u64) -> TabularData {
    let mut rng = CausalRng::from_seed(seed);
    let (mut a, mut b, mut c, mut d) = (vec![], vec![], vec![], vec![]);
    for _ in 0..n {
        let (x, y) = (normal(&mut rng), normal(&mut rng));
        let z = 0.8 * x + 0.8 * y + 0.6 * normal(&mut rng);
        a.push(x);
        b.push(y);
        c.push(z);
        d.push(0.9 * z + 0.5 * normal(&mut rng));
    }
    TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("b", b.as_slice()),
        ("c", c.as_slice()),
        ("d", d.as_slice()),
    ])
    .unwrap()
}

fn lag1_series(n: usize, seed: u64) -> TimeSeriesData {
    let mut rng = CausalRng::from_seed(seed);
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 1..n {
        x[t] = normal(&mut rng);
        y[t] = 0.8 * x[t - 1] + 0.2 * normal(&mut rng);
    }
    let data = TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap();
    TimeSeriesData::try_new(
        data.storage().clone(),
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap()
}

fn selective(
    execution: &CompiledExecution,
) -> antecedent::analysis::selective_discovery::SelectiveSearchResult {
    let Some(DiscoveredStructure::Selective(found)) =
        execution.result().discovery().map(|payload| payload.structure.clone())
    else {
        panic!("a selective search returns its selective outcome");
    };
    *found
}

#[test]
fn a_compiled_static_selective_search_is_the_adapter_fresh_and_resumed() {
    on_large_stack("twin-selective-static", || {
        let data = collider_table(300, 7);
        let contract = StaticDiscoveryContract::new(
            StaticAlgorithm::Pc,
            vars(4),
            CiTestSpec::named("parcorr"),
            SEED,
        );
        let spec = DiscoverySpec::selective_static(contract.clone()).unwrap();
        let study = Study::tabular(data.clone()).discovery(spec.clone(), vars(4)).build().unwrap();
        let compiled = selective(&CompiledExecution::run(&study, &ctx()).unwrap());

        let mut adapter = StaticSelectiveDiscovery::new();
        let report = adapter.run(&contract, &data, &ctx()).unwrap();
        let direct = adapter.outcome().unwrap();
        assert_eq!(compiled.outcome.result_identity, direct.result_identity);
        assert_eq!(compiled.outcome.structure_identity, direct.structure_identity);
        assert_eq!(format!("{:?}", compiled.outcome.structure), format!("{:?}", direct.structure));
        assert_eq!(format!("{:?}", compiled.report), format!("{report:?}"));
        let state = adapter.export_state().unwrap();
        assert_eq!(format!("{:?}", compiled.state), format!("{:?}", Some(state.clone())));

        // Resumed from the retained state: the adapter's resume, through the compiler.
        let resumed_spec = spec.clone().resuming(state.clone()).unwrap();
        assert_ne!(
            resumed_spec.query(vars(4)).unwrap(),
            spec.query(vars(4)).unwrap(),
            "a resumed search is another program"
        );
        let study = Study::tabular(data.clone()).discovery(resumed_spec, vars(4)).build().unwrap();
        let compiled = selective(&CompiledExecution::run(&study, &ctx()).unwrap());
        let (resumed, report) =
            StaticSelectiveDiscovery::resume(&contract, Some(&data), &state, &ctx()).unwrap();
        let direct = resumed.outcome().unwrap();
        assert_eq!(compiled.outcome.result_identity, direct.result_identity);
        assert_eq!(format!("{:?}", compiled.outcome.structure), format!("{:?}", direct.structure));
        assert_eq!(format!("{:?}", compiled.report), format!("{report:?}"));
    });
}

#[test]
fn a_compiled_temporal_selective_search_is_the_adapter() {
    on_large_stack("twin-selective-temporal", || {
        let series = lag1_series(240, 11);
        let contract = TemporalDiscoveryContract::new(
            TemporalAlgorithm::PcmciPlus,
            vars(2),
            CiTestSpec::named("parcorr"),
            SEED,
        );
        let spec = DiscoverySpec::selective_temporal(contract.clone()).unwrap();
        let study = Study::series(series.clone()).discovery(spec, vars(2)).build().unwrap();
        let compiled = selective(&CompiledExecution::run(&study, &ctx()).unwrap());

        let mut adapter = TemporalSelectiveDiscovery::new();
        let report = adapter.run(&contract, &series, &ctx()).unwrap();
        let direct = adapter.outcome().unwrap();
        assert_eq!(compiled.outcome.result_identity, direct.result_identity);
        assert_eq!(format!("{:?}", compiled.outcome.structure), format!("{:?}", direct.structure));
        assert_eq!(format!("{:?}", compiled.report), format!("{report:?}"));
    });
}

#[test]
fn a_selective_search_refuses_other_variables_and_only_a_selective_search_resumes() {
    on_large_stack("selective-refusals", || {
        let data = collider_table(120, 13);
        let contract = StaticDiscoveryContract::new(
            StaticAlgorithm::Pc,
            vars(4),
            CiTestSpec::named("parcorr"),
            SEED,
        );
        let mut adapter = StaticSelectiveDiscovery::new();
        adapter.run(&contract, &data, &ctx()).unwrap();
        let state = adapter.export_state().unwrap();

        let plain = DiscoverySpec::pc(
            StaticDiscoverParams {
                alpha: 0.05,
                max_cond_size: 2,
                fdr: None,
                ci: resolve_ci("parcorr", None).unwrap(),
                screen_pc: false,
                max_subset: None,
                knowledge: BackgroundKnowledge::default(),
            },
            "parcorr",
        );
        assert!(plain.resuming(state).is_err(), "only a selective search resumes");

        let spec = DiscoverySpec::selective_static(contract).unwrap();
        let reordered = vec![
            VariableId::from_raw(1),
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            VariableId::from_raw(3),
        ];
        let study = Study::tabular(data).discovery(spec, reordered).build().unwrap();
        let error = CompiledExecution::run(&study, &ctx()).unwrap_err();
        assert!(error.to_string().contains("invalid_argument"), "{error}");
    });
}

fn signed_uniform() -> BayesianDiscoverParams {
    BayesianDiscoverParams {
        prior: GraphPrior::default(),
        score_family: GraphScoreFamily::GaussianSignedUniform,
    }
}

fn observed(
    execution: &CompiledExecution,
) -> antecedent_discovery::observed_marginals::ObservedMarginals {
    let Some(DiscoveredStructure::ObservedMarginals(found)) =
        execution.result().discovery().map(|payload| payload.structure.clone())
    else {
        panic!("a draw-limited signed-uniform search returns observed marginals");
    };
    found
}

/// Three columns, the last missing on every fifth row.
fn partially_observed(n: usize, seed: u64) -> TabularData {
    let mut rng = CausalRng::from_seed(seed);
    let (mut x, mut z, mut y) = (vec![], vec![], vec![]);
    for row in 0..n {
        let xi = normal(&mut rng);
        let zi = 0.7 * xi + normal(&mut rng);
        let yi = 0.6 * zi + normal(&mut rng);
        x.push(xi);
        z.push(zi);
        y.push(if row % 5 == 4 { f64::NAN } else { yi });
    }
    TabularData::from_f64_columns([("x", x.as_slice()), ("z", z.as_slice()), ("y", y.as_slice())])
        .unwrap()
}

#[test]
fn compiled_observed_marginals_are_the_kernel_under_the_master_seed() {
    on_large_stack("twin-observed-marginals", || {
        let data = partially_observed(40, 17);
        let spec =
            DiscoverySpec::exact_dag_marginals(signed_uniform()).with_observed_marginal_draws(4096);
        assert_ne!(
            spec.query(vars(3)).unwrap(),
            DiscoverySpec::exact_dag_marginals(signed_uniform()).query(vars(3)).unwrap(),
            "a draw limit is part of the program"
        );
        let study = Study::tabular(data.clone()).discovery(spec, vars(3)).build().unwrap();
        let compiled = observed(&CompiledExecution::run(&study, &ctx()).unwrap());
        let direct = signed_uniform_observed_marginals(
            &data,
            &vars(3),
            &GraphPrior::default(),
            ctx().rng.master_seed(),
            4096,
            &ctx(),
        )
        .unwrap();
        let bits = |values: &[f64]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&compiled.edge_marginals), bits(&direct.edge_marginals));
        assert_eq!(compiled.log_evidence.to_bits(), direct.log_evidence.to_bits());
        assert_eq!(compiled.draws_used, direct.draws_used);
        assert_eq!(compiled.effective_draws.to_bits(), direct.effective_draws.to_bits());
        assert_eq!(compiled.max_edge_mcse.to_bits(), direct.max_edge_mcse.to_bits());
        assert!(compiled.draws_used > 0, "the missing cells were integrated");
    });
}

/// Independent check: `x = (0.8, -0.3, 1.25)`, `y = (-1.4, 0.1, missing)`. An independent
/// integration of the missing value and the coefficient in the joint density gives
/// `P(x -> y) = 0.38202614` and `P(y -> x) = 0.28189405`; the compiled marginals agree within
/// three times the 0.01 Monte Carlo error bound the schedule stops at.
#[test]
fn compiled_observed_marginals_agree_with_an_independent_integration() {
    on_large_stack("truth-observed-marginals", || {
        let data = TabularData::from_f64_columns([
            ("x", [0.8, -0.3, 1.25].as_slice()),
            ("y", [-1.4, 0.1, f64::NAN].as_slice()),
        ])
        .unwrap();
        let spec =
            DiscoverySpec::exact_dag_marginals(signed_uniform()).with_observed_marginal_draws(8192);
        let study = Study::tabular(data).discovery(spec, vars(2)).build().unwrap();
        let found = observed(&CompiledExecution::run(&study, &ctx()).unwrap());
        assert!(found.max_edge_mcse <= 0.01, "{found:?}");
        assert!((found.edge_marginals[1] - 0.382_026_14).abs() < 0.03, "{found:?}");
        assert!((found.edge_marginals[2] - 0.281_894_05).abs() < 0.03, "{found:?}");
    });
}

#[test]
fn a_draw_limit_on_another_search_is_refused() {
    on_large_stack("observed-refusal", || {
        let data = partially_observed(40, 19);
        let spec = DiscoverySpec::exact_dag_marginals(BayesianDiscoverParams::default())
            .with_observed_marginal_draws(256);
        let study = Study::tabular(data).discovery(spec, vars(3)).build().unwrap();
        let error = CompiledExecution::run(&study, &ctx()).unwrap_err();
        assert!(error.to_string().contains("invalid_argument"), "{error}");
    });
}
