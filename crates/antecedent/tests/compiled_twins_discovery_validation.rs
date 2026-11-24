//! Twin equality of the discovery validation checks and their compiled route, and the checks'
//! answers on a series whose lag-one link is known.
//!
//! Each twin runs the `antecedent-validate` check directly, with the engine configured the way
//! the direct Python validators configure it, and the compiled check
//! (`Study::validate_discovery`) on a study whose temporal search carries the same parameters,
//! and asserts that the reports are the same: every frequency, rate and standard error compared
//! by `to_bits()` and the whole report by its exact `Debug` text, which prints every float to
//! its shortest round-trip form. The compiled check runs the same kernel with the same engine
//! configuration, so any difference is a bug.
//!
//! Seeds: the compiled check hands the kernel the context it was run with; the direct call and
//! the check share one context seed, and no child seed is derived.
// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::float_cmp,
    clippy::too_many_lines,
    reason = "integration test with deterministic fixtures and bitwise comparisons"
)]

use std::sync::Arc;

use antecedent::analysis::discovery_validation::{DiscoveryCheck, DiscoveryCheckReport};
use antecedent::data::EnvHoldoutSplit;
use antecedent::discovery::{
    BackgroundKnowledge, DiscoverParams, DiscoveryWorkspace, JpcmciPlus, MultiDatasetConstraints,
    Pcmci, PcmciPlus, Rpcmci, two_regime_half_split,
};
use antecedent::discovery_defaults::{
    DEFAULT_MAX_COND_SIZE, contemporaneous_constraints, jpcmci_constraints, pcmci_constraints,
    resolve_ci,
};
use antecedent::validate::{
    AlphaThresholdSensitivity, BlockBootstrapStability, CiTestSensitivity, EnvironmentHoldout,
    FalsePositiveCheck, LagWindowSensitivity, NullTransform, OrientationStability, RegimeStability,
    SyntheticNullCalibration,
};
use antecedent::{DiscoverySpec, Study};
use antecedent_core::{CausalRng, ExecutionContext, VariableId};
use antecedent_data::{
    MultiEnvironmentData, SamplingRegularity, TabularData, TimeIndex, TimeSeriesData,
};

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(0x5EED_0077)
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

/// `x(t-1) -> y(t)` with coefficient 0.8 and noise scale 0.2; `x` is white noise.
fn lag1_series(n: usize, seed: u64) -> TimeSeriesData {
    let mut rng = CausalRng::from_seed(seed);
    let mut normal = || {
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    };
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 1..n {
        x[t] = normal();
        y[t] = 0.8 * x[t - 1] + 0.2 * normal();
    }
    let data = TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap();
    TimeSeriesData::try_new(
        data.storage().clone(),
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap()
}

fn vars() -> Vec<VariableId> {
    vec![v(0), v(1)]
}

/// The parameters the direct Python validators fix: the default conditioning-set bound, no
/// background knowledge and no multi-dataset constraints.
fn params(max_lag: u32, alpha: f64, fdr: bool, ci: &str) -> DiscoverParams {
    DiscoverParams {
        max_lag,
        alpha,
        fdr: fdr.then(antecedent::inference::FdrAdjustment::bh),
        ci: resolve_ci(ci, None).unwrap(),
        multi_dataset: MultiDatasetConstraints::default(),
        max_cond_size: DEFAULT_MAX_COND_SIZE,
        knowledge: BackgroundKnowledge::default(),
    }
}

/// PCMCI as the direct validators build it.
fn direct_pcmci(max_lag: u32, alpha: f64, fdr: bool, ci: &str) -> Pcmci {
    Pcmci::new()
        .with_fdr(fdr)
        .with_constraints(pcmci_constraints(max_lag, alpha, DEFAULT_MAX_COND_SIZE))
        .with_ci(resolve_ci(ci, None).unwrap())
}

/// PCMCI+ as the direct validators build it.
fn direct_pcmci_plus(max_lag: u32, alpha: f64, fdr: bool, ci: &str) -> PcmciPlus {
    PcmciPlus::new()
        .with_fdr(fdr)
        .with_constraints(contemporaneous_constraints(max_lag, alpha, DEFAULT_MAX_COND_SIZE))
        .with_ci(resolve_ci(ci, None).unwrap())
}

fn series_study(series: &TimeSeriesData, spec: DiscoverySpec) -> Study {
    Study::series(series.clone()).discovery(spec, vars()).build().unwrap()
}

fn compiled(study: &Study, check: DiscoveryCheck) -> DiscoveryCheckReport {
    let validation = study.validate_discovery(check, &ctx()).unwrap();
    assert_eq!(validation.lowered().route(), "DiscoveryValidation");
    validation.report().clone()
}

fn bits(values: impl IntoIterator<Item = f64>) -> Vec<u64> {
    values.into_iter().map(f64::to_bits).collect()
}

fn stability(report: DiscoveryCheckReport) -> antecedent::validate::DiscoveryStabilityReport {
    let DiscoveryCheckReport::Stability(report) = report else {
        panic!("a link-frequency check reports link frequencies");
    };
    report
}

#[test]
fn pcmci_checks_match_their_direct_validators() {
    on_large_stack("twin-pcmci-checks", || {
        let series = lag1_series(240, 41);
        let study = series_study(
            &series,
            DiscoverySpec::pcmci(params(1, 0.05, false, "parcorr"), "parcorr"),
        );
        let mut ws = DiscoveryWorkspace::default();

        let direct = BlockBootstrapStability {
            pcmci: direct_pcmci(1, 0.05, false, "parcorr"),
            replicates: 12,
            block_size: 20,
        }
        .run(&series, &vars(), &mut ws, &ctx())
        .unwrap();
        let twin = stability(compiled(
            &study,
            DiscoveryCheck::BlockBootstrap { replicates: 12, block_size: 20 },
        ));
        assert_eq!(
            bits(twin.frequencies.iter().map(|l| l.frequency)),
            bits(direct.frequencies.iter().map(|l| l.frequency))
        );
        assert_eq!(format!("{twin:?}"), format!("{direct:?}"));

        for transform in [NullTransform::ColumnPermute, NullTransform::PhaseRandomize] {
            let direct =
                FalsePositiveCheck::new(direct_pcmci(1, 0.05, false, "parcorr"), transform, 10)
                    .run(&series, &vars(), &mut ws, &ctx())
                    .unwrap();
            let DiscoveryCheckReport::FalsePositive(twin) =
                compiled(&study, DiscoveryCheck::FalsePositive { transform, replicates: 10 })
            else {
                panic!("the false-positive check reports a rate");
            };
            assert_eq!(twin.empirical_fpr.to_bits(), direct.empirical_fpr.to_bits());
            assert_eq!(twin.se.to_bits(), direct.se.to_bits());
            assert_eq!(twin.mean_edge_count.to_bits(), direct.mean_edge_count.to_bits());
            assert_eq!(format!("{twin:?}"), format!("{direct:?}"));
        }

        // A grid's study declares the grid's first point, as the direct validators do.
        let alphas: Arc<[f64]> = Arc::from([0.01, 0.05, 0.2]);
        let grid_study = series_study(
            &series,
            DiscoverySpec::pcmci(params(1, 0.01, false, "parcorr"), "parcorr"),
        );
        let direct = AlphaThresholdSensitivity::new(
            direct_pcmci(1, 0.01, false, "parcorr"),
            Arc::clone(&alphas),
        )
        .run(&series, &vars(), &mut ws, &ctx())
        .unwrap();
        let twin = stability(compiled(&grid_study, DiscoveryCheck::Alpha { alphas }));
        assert_eq!(format!("{twin:?}"), format!("{direct:?}"));

        let max_lags: Arc<[u32]> = Arc::from([1, 2, 3]);
        let direct = LagWindowSensitivity::new(
            direct_pcmci(1, 0.05, false, "parcorr"),
            Arc::clone(&max_lags),
        )
        .run(&series, &vars(), &mut ws, &ctx())
        .unwrap();
        let twin = stability(compiled(&study, DiscoveryCheck::LagWindow { max_lags }));
        assert_eq!(format!("{twin:?}"), format!("{direct:?}"));

        let ci_names: Arc<[Arc<str>]> =
            Arc::from([Arc::from("parcorr"), Arc::from("robust_parcorr")]);
        let direct =
            CiTestSensitivity::new(direct_pcmci(1, 0.05, false, "parcorr"), Arc::clone(&ci_names))
                .run(&series, &vars(), &mut ws, &ctx())
                .unwrap();
        let twin = stability(compiled(&study, DiscoveryCheck::CiTest { ci_names }));
        assert_eq!(format!("{twin:?}"), format!("{direct:?}"));

        let direct =
            SyntheticNullCalibration::new(direct_pcmci(1, 0.05, false, "parcorr"), 0.05, 8, 120, 3)
                .run(&mut ws, &ctx())
                .unwrap();
        let DiscoveryCheckReport::NullCalibration(twin) =
            compiled(&study, DiscoveryCheck::SyntheticNull { n_sim: 8, n_obs: 120, n_vars: 3 })
        else {
            panic!("the synthetic-null check reports a rate");
        };
        assert_eq!(twin.empirical_fpr.to_bits(), direct.empirical_fpr.to_bits());
        assert_eq!(twin.se.to_bits(), direct.se.to_bits());
        assert_eq!(format!("{twin:?}"), format!("{direct:?}"));
    });
}

#[test]
fn the_fdr_search_matches_its_direct_validator() {
    on_large_stack("twin-fdr-check", || {
        let series = lag1_series(200, 43);
        let study =
            series_study(&series, DiscoverySpec::pcmci(params(2, 0.1, true, "parcorr"), "parcorr"));
        let direct = BlockBootstrapStability {
            pcmci: direct_pcmci(2, 0.1, true, "parcorr"),
            replicates: 8,
            block_size: 25,
        }
        .run(&series, &vars(), &mut DiscoveryWorkspace::default(), &ctx())
        .unwrap();
        let twin = stability(compiled(
            &study,
            DiscoveryCheck::BlockBootstrap { replicates: 8, block_size: 25 },
        ));
        assert_eq!(format!("{twin:?}"), format!("{direct:?}"));
    });
}

#[test]
fn orientation_regime_and_holdout_checks_match_their_direct_validators() {
    on_large_stack("twin-plus-checks", || {
        let series = lag1_series(240, 47);
        let mut ws = DiscoveryWorkspace::default();

        let plus = series_study(
            &series,
            DiscoverySpec::pcmci_plus(params(1, 0.05, false, "parcorr"), "parcorr"),
        );
        let direct = OrientationStability {
            pcmci_plus: direct_pcmci_plus(1, 0.05, false, "parcorr"),
            replicates: 10,
            block_size: 20,
        }
        .run(&series, &vars(), &mut ws, &ctx())
        .unwrap();
        let DiscoveryCheckReport::Orientation(twin) =
            compiled(&plus, DiscoveryCheck::Orientation { replicates: 10, block_size: 20 })
        else {
            panic!("the orientation check reports orientation frequencies");
        };
        assert_eq!(twin.conflict_rate.to_bits(), direct.conflict_rate.to_bits());
        assert_eq!(format!("{twin:?}"), format!("{direct:?}"));

        let regimes = two_regime_half_split(series.row_count());
        let regime_study = series_study(
            &series,
            DiscoverySpec::rpcmci(
                params(1, 0.05, false, "parcorr"),
                regimes.clone(),
                None,
                "parcorr",
            ),
        );
        let rpcmci = Rpcmci::new()
            .with_pcmci_plus(direct_pcmci_plus(1, 0.05, false, "parcorr"))
            .with_alternating_iters(0);
        let mut checker = RegimeStability::new(rpcmci, regimes);
        checker.replicates = 6;
        checker.block_size = 20;
        let direct = checker.run(&series, &vars(), &mut ws, &ctx()).unwrap();
        let DiscoveryCheckReport::RegimeStability(twin) = compiled(
            &regime_study,
            DiscoveryCheck::RegimeStability { replicates: 6, block_size: 20 },
        ) else {
            panic!("the regime check reports per-regime frequencies");
        };
        assert_eq!(format!("{twin:?}"), format!("{direct:?}"));

        let environments = MultiEnvironmentData::try_new(Arc::from([
            lag1_series(160, 51),
            lag1_series(160, 52),
            lag1_series(160, 53),
        ]))
        .unwrap();
        let joint = Study::series_multi(environments.clone())
            .discovery(
                DiscoverySpec::jpcmci_plus(params(1, 0.05, false, "parcorr"), "parcorr"),
                vars(),
            )
            .build()
            .unwrap();
        let jpcmci = JpcmciPlus::new()
            .with_fdr(false)
            .with_constraints(jpcmci_constraints(
                1,
                0.05,
                DEFAULT_MAX_COND_SIZE,
                MultiDatasetConstraints::default(),
            ))
            .with_ci(resolve_ci("parcorr", None).unwrap());
        let split = EnvHoldoutSplit::try_prefix(environments.env_count(), 2).unwrap();
        let direct = EnvironmentHoldout::new(jpcmci, split)
            .run(&environments, &vars(), &mut ws, &ctx())
            .unwrap();
        let DiscoveryCheckReport::EnvironmentHoldout(twin) =
            compiled(&joint, DiscoveryCheck::EnvironmentHoldout { n_discovery: 2 })
        else {
            panic!("the holdout check reports link agreement");
        };
        assert_eq!(twin.jaccard.to_bits(), direct.jaccard.to_bits());
        assert_eq!(format!("{twin:?}"), format!("{direct:?}"));
    });
}

/// Known truth: the lag-one link `x(t-1) -> y(t)` (coefficient 0.8, noise 0.2, 400 rows) is
/// selected by PCMCI on every block resample and at every level of the grid, and no other link
/// is selected at every level; the synthetic-null check runs at the study's own level.
#[test]
fn the_checks_find_the_known_lag_one_link() {
    on_large_stack("truth-checks", || {
        let series = lag1_series(400, 59);
        let study = series_study(
            &series,
            DiscoverySpec::pcmci(params(1, 0.01, false, "parcorr"), "parcorr"),
        );
        let is_true_link = |link: &antecedent::discovery::LaggedLink| {
            link.source == v(0) && link.target == v(1) && link.source_lag.raw() == 1
        };

        let boot = stability(compiled(
            &study,
            DiscoveryCheck::BlockBootstrap { replicates: 20, block_size: 40 },
        ));
        let found = boot.frequencies.iter().find(|l| is_true_link(&l.link)).expect("the link");
        assert_eq!(found.frequency, 1.0, "the true link is selected on every resample");

        let grid = stability(compiled(
            &study,
            DiscoveryCheck::Alpha { alphas: Arc::from([0.001, 0.01, 0.05]) },
        ));
        let found = grid.frequencies.iter().find(|l| is_true_link(&l.link)).expect("the link");
        assert_eq!(found.frequency, 1.0, "the true link is selected at every level");
        assert!(
            grid.frequencies.iter().filter(|l| l.frequency == 1.0).all(|l| is_true_link(&l.link)),
            "only the true link is selected at every level: {grid:?}"
        );

        let DiscoveryCheckReport::NullCalibration(null) =
            compiled(&study, DiscoveryCheck::SyntheticNull { n_sim: 10, n_obs: 150, n_vars: 3 })
        else {
            panic!("the synthetic-null check reports a rate");
        };
        assert_eq!(null.alpha, 0.01, "the null runs at the study's level");
        assert_eq!(null.n_sim, 10);
    });
}

#[test]
fn a_check_refuses_a_search_it_does_not_validate() {
    on_large_stack("check-refusals", || {
        let series = lag1_series(160, 61);
        let plus = series_study(
            &series,
            DiscoverySpec::pcmci_plus(params(1, 0.05, false, "parcorr"), "parcorr"),
        );
        let error = plus
            .validate_discovery(
                DiscoveryCheck::BlockBootstrap { replicates: 4, block_size: 20 },
                &ctx(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("route_not_supported"), "{error}");

        let pcmci = series_study(
            &series,
            DiscoverySpec::pcmci(params(1, 0.05, false, "parcorr"), "parcorr"),
        );
        let error = pcmci
            .validate_discovery(
                DiscoveryCheck::Orientation { replicates: 4, block_size: 20 },
                &ctx(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("route_not_supported"), "{error}");
        assert!(error.to_string().contains("discovery_validation.wrong_search"), "{error}");

        let error = Study::series(series.clone())
            .discovery(DiscoverySpec::pcmci(params(1, 0.05, false, "parcorr"), "parcorr"), vars())
            .build()
            .unwrap()
            .validate_discovery(DiscoveryCheck::EnvironmentHoldout { n_discovery: 1 }, &ctx())
            .unwrap_err();
        assert!(error.to_string().contains("discovery_validation.wrong_search"), "{error}");
    });
}

/// The check's program is keyed by the search program and by every option of the check: the same
/// check on the same study is the same program, another replicate count is another program, and
/// the program names the DiscoveryValidation route.
#[test]
fn a_check_program_is_keyed_by_the_search_and_its_options() {
    on_large_stack("check-identity", || {
        let series = lag1_series(160, 61);
        let pcmci = series_study(
            &series,
            DiscoverySpec::pcmci(params(1, 0.05, false, "parcorr"), "parcorr"),
        );
        let four = pcmci
            .validate_discovery(
                DiscoveryCheck::BlockBootstrap { replicates: 4, block_size: 20 },
                &ctx(),
            )
            .unwrap();
        let five = pcmci
            .validate_discovery(
                DiscoveryCheck::BlockBootstrap { replicates: 5, block_size: 20 },
                &ctx(),
            )
            .unwrap();
        assert_ne!(four.lowered_identity(), five.lowered_identity());
        let again = pcmci
            .validate_discovery(
                DiscoveryCheck::BlockBootstrap { replicates: 4, block_size: 20 },
                &ctx(),
            )
            .unwrap();
        assert_eq!(four.lowered_identity(), again.lowered_identity());
        assert_eq!(four.lowered().route(), "DiscoveryValidation");
        let other_search = series_study(
            &series,
            DiscoverySpec::pcmci(params(1, 0.1, false, "parcorr"), "parcorr"),
        )
        .validate_discovery(
            DiscoveryCheck::BlockBootstrap { replicates: 4, block_size: 20 },
            &ctx(),
        )
        .unwrap();
        assert_ne!(four.lowered_identity(), other_search.lowered_identity());
    });
}
