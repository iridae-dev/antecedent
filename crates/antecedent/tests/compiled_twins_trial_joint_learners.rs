//! The compiled caller-learner trial AIPW transport and factorized joint-cell fit reach the same
//! factory kernels as the direct entries.
//!
//! - `TrialAipwLearnerTransport`: `Study::compile_trial_aipw_learners` decides the transport by
//!   sID and validates the request; `execute` calls `estimate_trial_aipw_with_factories` with the
//!   declared learners and the caller's context. The twin compares the point, every score, every
//!   out-of-fold nuisance and the whole estimate with the direct call bit for bit. The known-truth
//!   check hands the library's own learners over as caller factories on a noise-free trial whose
//!   target contrast is exactly 2, and checks that the compiled point is 2 and equals the
//!   library-learner route.
//! - `JointCellsLearnerFit`: `Study::compile_joint_cells_learners` binds the joint query;
//!   `execute` calls `fit_factorized_joint_cells_with_factories`. The twin compares every cell,
//!   score, propensity, normalization check and ordering spread with the direct call. The
//!   known-truth check draws a two-component joint law with cell means `0.4 c + 1{c = 3} + 0.25`
//!   and correctly specified nuisances, and checks the compiled cells against it and against the
//!   built-in declared-learner route.
//!
//! Both routes are point-only and report calibration `not_measured`; each run carries the
//! calibration label of its declared learners, which differs between declarations, so coverage
//! measured with the library learners never annotates a factory fit.
//!
//! Seeds: both kernels draw only from the caller's context (fold plans are keyed by the rows or
//! the declared seed), and the compiled route passes the same context scoped to its program, so
//! no seed difference arises.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "the bits are compared exactly")]

use std::sync::Arc;

use antecedent::Study;
use antecedent::analysis::declared_learners::LearnerFactories;
use antecedent_core::ExecutionContext;
use antecedent_estimate::{LearnerSpec, LinearSpec, LogisticSpec};
use antecedent_learn::{LearnerFactory, PredictionTask, resolve_for};

/// The library's linear outcome and logistic membership learners, handed over as caller
/// factories under `identity`.
fn library_learners(
    identity: &str,
) -> (Arc<dyn LearnerFactory>, Arc<dyn LearnerFactory>, LearnerFactories) {
    let outcome: Arc<dyn LearnerFactory> = Arc::from(
        resolve_for(LearnerSpec::Linear(LinearSpec::default()), PredictionTask::Regression)
            .unwrap(),
    );
    let membership: Arc<dyn LearnerFactory> = Arc::from(
        resolve_for(
            LearnerSpec::Logistic(LogisticSpec::default()),
            PredictionTask::BinaryProbability,
        )
        .unwrap(),
    );
    let learners =
        LearnerFactories::new(Arc::clone(&outcome), Arc::clone(&membership), identity).unwrap();
    (outcome, membership, learners)
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

// ---- trial AIPW transport with caller learners ------------------------------------------------

mod trial {
    use super::*;

    use antecedent::analysis::compiled_trial_aipw_learners::TrialAipwLearnersRequest;
    use antecedent_core::{
        ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery, TransportQuery, VariableId,
    };
    use antecedent_estimate::{
        TrialAipwEstimate, TrialAipwInput, TrialAipwOptions, TrialSampling, estimate_trial_aipw,
        estimate_trial_aipw_with_factories,
    };
    use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
    use antecedent_identify::TransportIdentifier;

    const IDENTITY: &str = "outcome=linear;membership=logistic";

    fn diagram() -> SelectionDiagram {
        let mut graph = Admg::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        SelectionDiagram::try_new(graph, vec![]).unwrap()
    }

    fn query() -> TransportQuery {
        TransportQuery::new(
            ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome: VariableId::from_raw(1),
                treatment: ContinuousDomain::new(
                    VariableId::from_raw(0),
                    GridSpec::Values(Arc::from([0., 1.])),
                ),
            }),
            "trial",
            "target",
            [VariableId::from_raw(0)],
        )
    }

    /// A randomized source trial (rows 0..120) whose outcome is `1 + 2 t` exactly, and target
    /// nonparticipants (rows 120..200): the transported contrast is exactly 2.
    fn input(sampling: TrialSampling) -> TrialAipwInput {
        TrialAipwInput {
            features: vec![],
            covariates: vec![],
            outcome: (0..200)
                .map(|i| if i < 120 { 1. + 2. * f64::from(i % 2) } else { 0. })
                .collect(),
            treatment: (0..200).map(|i| i % 2 == 1).collect(),
            source: (0..200).map(|i| i < 120).collect(),
            randomization: vec![0.5; 200],
            sampling,
        }
    }

    fn options() -> TrialAipwOptions {
        TrialAipwOptions {
            outcome: LearnerSpec::Linear(LinearSpec::default()),
            membership: LearnerSpec::Logistic(LogisticSpec::default()),
            folds: 3,
            bootstrap: 0,
            ..TrialAipwOptions::default()
        }
    }

    fn request(identity: &str, input: TrialAipwInput) -> TrialAipwLearnersRequest {
        let (_, _, learners) = library_learners(identity);
        TrialAipwLearnersRequest::new(diagram(), query(), input, options(), learners)
    }

    fn assert_twin(label: &str, direct: &TrialAipwEstimate, compiled: &TrialAipwEstimate) {
        assert_eq!(direct.estimate.to_bits(), compiled.estimate.to_bits(), "{label}: point");
        assert_eq!(
            direct.interval.map(|(lo, hi)| (lo.to_bits(), hi.to_bits())),
            compiled.interval.map(|(lo, hi)| (lo.to_bits(), hi.to_bits())),
            "{label}: interval"
        );
        assert_eq!(bits(&direct.scores), bits(&compiled.scores), "{label}: scores");
        assert_eq!(bits(&direct.membership), bits(&compiled.membership), "{label}: membership");
        assert_eq!(bits(&direct.mu0), bits(&compiled.mu0), "{label}: mu0");
        assert_eq!(bits(&direct.mu1), bits(&compiled.mu1), "{label}: mu1");
        assert_eq!(
            serde_json::to_value(direct).unwrap(),
            serde_json::to_value(compiled).unwrap(),
            "{label}: every other field"
        );
    }

    #[test]
    fn the_compiled_trial_aipw_with_caller_learners_equals_the_factory_kernel() {
        for sampling in [TrialSampling::IndependentSamples, TrialSampling::NestedCohort] {
            let label = format!("{sampling:?}");
            let rows = input(sampling);
            let (outcome, membership, learners) = library_learners(IDENTITY);
            let id = TransportIdentifier::new().identify(&diagram(), &query()).unwrap();
            let direct = estimate_trial_aipw_with_factories(
                &id,
                &rows,
                &options(),
                &ExecutionContext::for_tests(11),
                &*outcome,
                &*membership,
            )
            .expect(&label);
            let ctx = ExecutionContext::for_tests(11);
            let compiled = Study::compile_trial_aipw_learners(
                TrialAipwLearnersRequest::new(diagram(), query(), rows, options(), learners),
                &ctx,
            )
            .expect(&label);
            assert_eq!(compiled.plan_view().route().as_str(), "TrialAipwLearnerTransport");
            assert_eq!(compiled.identification(), &id, "{label}: the same certificate");
            let run = compiled.execute(&ctx).expect(&label);
            assert_twin(&label, &direct, &run.estimate);
            assert!(run.estimate.external_nuisances_unverified, "{label}");
            assert_eq!((run.claim, run.calibration), ("point_only", "not_measured"));
        }
    }

    #[test]
    fn the_compiled_trial_aipw_with_the_library_learners_recovers_the_known_contrast() {
        let ctx = ExecutionContext::for_tests(11);
        let run = Study::compile_trial_aipw_learners(
            request(IDENTITY, input(TrialSampling::IndependentSamples)),
            &ctx,
        )
        .unwrap()
        .execute(&ctx)
        .unwrap();
        assert!((run.estimate.estimate - 2.0).abs() < 1e-10, "{}", run.estimate.estimate);
        // The library-learner route on the same rows: the factories are those learners.
        let id = TransportIdentifier::new().identify(&diagram(), &query()).unwrap();
        let library = estimate_trial_aipw(
            &id,
            &input(TrialSampling::IndependentSamples),
            &options(),
            &ExecutionContext::for_tests(11),
        )
        .unwrap();
        assert!((run.estimate.estimate - library.estimate).abs() < 1e-12);
        assert!(!library.external_nuisances_unverified);
    }

    #[test]
    fn caller_learners_with_bootstrap_replicates_refuse_through_the_compiled_route() {
        let ctx = ExecutionContext::for_tests(11);
        let (_, _, learners) = library_learners(IDENTITY);
        let replicated = TrialAipwOptions { bootstrap: 5, ..options() };
        let error = Study::compile_trial_aipw_learners(
            TrialAipwLearnersRequest::new(
                diagram(),
                query(),
                input(TrialSampling::IndependentSamples),
                replicated,
                learners,
            ),
            &ctx,
        )
        .unwrap()
        .execute(&ctx)
        .unwrap_err();
        assert_eq!(error.reason_code(), Some("route_not_supported"), "{error}");
    }

    #[test]
    fn the_trial_aipw_program_identity_binds_the_rows_the_options_and_the_learners() {
        let ctx = ExecutionContext::for_tests(11);
        let compile = |request: TrialAipwLearnersRequest| {
            Study::compile_trial_aipw_learners(request, &ctx).unwrap()
        };
        let rows = || input(TrialSampling::IndependentSamples);
        let base = compile(request(IDENTITY, rows())).lowered().identity();
        assert_eq!(base, compile(request(IDENTITY, rows())).lowered().identity(), "replayable");
        assert_ne!(base, compile(request("other learners", rows())).lowered().identity());
        let mut changed = rows();
        changed.outcome[3] = f64::from_bits(changed.outcome[3].to_bits() ^ 1);
        assert_ne!(base, compile(request(IDENTITY, changed)).lowered().identity(), "rows");
        let (_, _, learners) = library_learners(IDENTITY);
        let folds = TrialAipwOptions { folds: 4, ..options() };
        assert_ne!(
            base,
            compile(TrialAipwLearnersRequest::new(diagram(), query(), rows(), folds, learners))
                .lowered()
                .identity(),
            "options"
        );
    }

    #[test]
    fn a_trial_factory_fit_carries_the_calibration_label_of_its_declared_learners() {
        let ctx = ExecutionContext::for_tests(11);
        let label = |identity: &str| {
            Study::compile_trial_aipw_learners(
                request(identity, input(TrialSampling::IndependentSamples)),
                &ctx,
            )
            .unwrap()
            .execute(&ctx)
            .unwrap()
            .calibration_label
        };
        let declared = label(IDENTITY);
        assert!(declared.starts_with("learners=caller_factory:"), "{declared}");
        assert_ne!(declared, label("other learners"));
    }
}

// ---- factorized joint cells with caller learners ----------------------------------------------

mod joint_cells {
    use super::*;

    use antecedent::analysis::compiled_joint_cells_learners::JointCellsLearnersRequest;
    use antecedent_core::{StreamDomain, VariableId};
    use antecedent_data::{TableView, TabularData};
    use antecedent_estimate::{
        CellStatus, FactorizedJointConfig, FactorizedJointFit, RidgeTuning,
        fit_factorized_joint_cells, fit_factorized_joint_cells_with_factories, orderings_for,
    };

    const IDENTITY: &str = "outcome=linear;propensity=logistic";

    /// `P(cell | z)` for two binary components (cell bit `j` is `t_j`), every cell populated.
    const LAW: [[f64; 4]; 2] = [[0.4, 0.2, 0.2, 0.2], [0.1, 0.3, 0.2, 0.4]];

    /// `z ~ Bernoulli(1/2)`, the cell drawn from `LAW[z]`, and
    /// `Y = 0.4 cell + 1{cell = 3} + 0.5 z + U(-0.3, 0.3)`, so
    /// `E[Y^{do(cell)}] = 0.4 cell + 1{cell = 3} + 0.25`.
    #[allow(clippy::cast_precision_loss, reason = "cell codes are below four")]
    fn table(n: usize, seed: u64) -> TabularData {
        let mut rng =
            ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0xE5);
        let (mut t0, mut t1, mut y, mut z) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for _ in 0..n {
            let level = usize::from(rng.next_f64() < 0.5);
            let u = rng.next_f64();
            let mut cumulative = 0.0;
            let mut cell = 3;
            for (c, p) in LAW[level].iter().enumerate() {
                cumulative += p;
                if u < cumulative {
                    cell = c;
                    break;
                }
            }
            let noise = 0.6 * (rng.next_f64() - 0.5);
            t0.push(f64::from(u8::from(cell & 1 == 1)));
            t1.push(f64::from(u8::from(cell & 2 == 2)));
            z.push(f64::from(u8::from(level == 1)));
            y.push(
                0.4 * cell as f64
                    + f64::from(u8::from(cell == 3))
                    + 0.5 * f64::from(u8::from(level == 1))
                    + noise,
            );
        }
        TabularData::from_f64_columns([
            ("t0", t0.as_slice()),
            ("t1", t1.as_slice()),
            ("y", y.as_slice()),
            ("z", z.as_slice()),
        ])
        .unwrap()
    }

    fn id(data: &TabularData, name: &str) -> VariableId {
        data.schema().id_of(name).unwrap()
    }

    fn config() -> FactorizedJointConfig {
        let mut config = FactorizedJointConfig::new(RidgeTuning::default());
        config.seed = 7;
        config
    }

    fn orderings() -> Vec<Vec<usize>> {
        orderings_for(2, &[0, 1], true).unwrap()
    }

    fn request(identity: &str, data: TabularData) -> JointCellsLearnersRequest {
        let (_, _, learners) = library_learners(identity);
        let treatments = vec![id(&data, "t0"), id(&data, "t1")];
        let (outcome, adjustment) = (id(&data, "y"), vec![id(&data, "z")]);
        JointCellsLearnersRequest::new(
            data,
            treatments,
            outcome,
            adjustment,
            orderings(),
            config(),
            learners,
        )
    }

    fn estimate_of(fit: &FactorizedJointFit, cell: usize) -> f64 {
        match &fit.cells[cell].status {
            CellStatus::Supported(estimate) => estimate.estimate,
            CellStatus::Unsupported(refusal) => panic!("cell {cell} unsupported: {refusal:?}"),
        }
    }

    fn assert_twin(direct: &FactorizedJointFit, compiled: &FactorizedJointFit) {
        assert_eq!(direct.cells.len(), compiled.cells.len());
        for cell in 0..direct.cells.len() {
            assert_eq!(
                estimate_of(direct, cell).to_bits(),
                estimate_of(compiled, cell).to_bits(),
                "cell {cell}: point"
            );
        }
        assert_eq!(direct.cells, compiled.cells, "every cell report");
        assert_eq!(bits(&direct.scores.scores), bits(&compiled.scores.scores), "scores");
        assert_eq!(
            bits(&direct.scores.propensities),
            bits(&compiled.scores.propensities),
            "propensities"
        );
        assert_eq!(
            direct.scores.nuisance_provenance, compiled.scores.nuisance_provenance,
            "score provenance"
        );
        assert_eq!(direct.normalization, compiled.normalization, "normalization");
        assert_eq!(direct.sensitivity, compiled.sensitivity, "ordering sensitivity");
        assert_eq!(
            (direct.n_rows, direct.folds, direct.degenerate_conditionals),
            (compiled.n_rows, compiled.folds, compiled.degenerate_conditionals)
        );
        assert_eq!(direct.treatments, compiled.treatments);
    }

    #[test]
    fn the_compiled_joint_cells_with_caller_learners_equal_the_factory_kernel() {
        let data = table(2000, 31);
        let (outcome, propensity, _) = library_learners(IDENTITY);
        let treatments = vec![id(&data, "t0"), id(&data, "t1")];
        let direct = fit_factorized_joint_cells_with_factories(
            &data,
            &treatments,
            id(&data, "y"),
            &[id(&data, "z")],
            &orderings(),
            &config(),
            &ExecutionContext::for_tests(17),
            &*outcome,
            &*propensity,
        )
        .unwrap();
        let ctx = ExecutionContext::for_tests(17);
        let compiled = Study::compile_joint_cells_learners(request(IDENTITY, data)).unwrap();
        assert_eq!(compiled.plan_view().route().as_str(), "JointCellsLearnerFit");
        let run = compiled.execute(&ctx).unwrap();
        assert_twin(&direct, &run.fit);
        assert!(run.fit.external_nuisances_unverified);
        assert!(run.fit.scores.nuisance_provenance.contains("trust=unverified"));
        assert_eq!((run.claim, run.calibration), ("point_only", "not_measured"));
    }

    /// Both nuisances are correctly specified (`z` is binary, so a logistic in `z` is saturated
    /// in each prefix stratum, and each cell's outcome is linear in `z`): the AIPW cell mean has
    /// standard error near `0.17 / sqrt(n p_cell)`, below 0.006 at n = 6 000 with
    /// `p_cell >= 0.15`, so 0.03 is several standard errors and claims no rate.
    #[test]
    fn the_compiled_joint_cells_with_the_library_learners_recover_the_known_cell_means() {
        let data = table(6000, 41);
        let ctx = ExecutionContext::for_tests(17);
        let run = Study::compile_joint_cells_learners(request(IDENTITY, data.clone()))
            .unwrap()
            .execute(&ctx)
            .unwrap();
        for cell in 0..4_u8 {
            let truth = 0.4 * f64::from(cell) + f64::from(u8::from(cell == 3)) + 0.25;
            let estimate = estimate_of(&run.fit, usize::from(cell));
            assert!((estimate - truth).abs() < 0.03, "cell {cell}: {estimate} vs {truth}");
        }
        // The built-in declared-learner route fits the same linear and logistic learners.
        let mut built_in = config();
        built_in.learner = Some(LearnerSpec::Linear(LinearSpec::default()));
        let library = fit_factorized_joint_cells(
            &data,
            &[id(&data, "t0"), id(&data, "t1")],
            id(&data, "y"),
            &[id(&data, "z")],
            &orderings(),
            &built_in,
            &ExecutionContext::for_tests(17),
        )
        .unwrap();
        assert_eq!(run.fit.scores.scores.len(), library.scores.scores.len());
        for (a, b) in run.fit.scores.scores.iter().zip(&library.scores.scores) {
            assert!((a - b).abs() < 1e-10, "{a} vs {b}");
        }
    }

    #[test]
    fn a_built_in_learner_beside_caller_learners_refuses_through_the_compiled_route() {
        let data = table(400, 31);
        let (_, _, learners) = library_learners(IDENTITY);
        let mut declared = config();
        declared.learner = Some(LearnerSpec::Linear(LinearSpec::default()));
        let treatments = vec![id(&data, "t0"), id(&data, "t1")];
        let (outcome, adjustment) = (id(&data, "y"), vec![id(&data, "z")]);
        let error = Study::compile_joint_cells_learners(JointCellsLearnersRequest::new(
            data,
            treatments,
            outcome,
            adjustment,
            orderings(),
            declared,
            learners,
        ))
        .unwrap()
        .execute(&ExecutionContext::for_tests(17))
        .unwrap_err();
        assert!(error.to_string().contains("caller factories cannot be combined"), "{error}");
    }

    #[test]
    fn the_joint_cells_program_identity_binds_the_rows_the_query_and_the_learners() {
        let identity = |request: JointCellsLearnersRequest| {
            Study::compile_joint_cells_learners(request).unwrap().lowered().identity()
        };
        let data = table(400, 31);
        let base = identity(request(IDENTITY, data.clone()));
        assert_eq!(base, identity(request(IDENTITY, data.clone())), "replayable");
        assert_ne!(base, identity(request("other learners", data.clone())), "learners");
        assert_ne!(base, identity(request(IDENTITY, table(400, 32))), "rows");
        let (_, _, learners) = library_learners(IDENTITY);
        let treatments = vec![id(&data, "t0"), id(&data, "t1")];
        let (outcome, adjustment) = (id(&data, "y"), vec![id(&data, "z")]);
        let mut seeded = config();
        seeded.seed = 8;
        assert_ne!(
            base,
            identity(JointCellsLearnersRequest::new(
                data.clone(),
                treatments.clone(),
                outcome,
                adjustment.clone(),
                orderings(),
                seeded,
                learners.clone(),
            )),
            "configuration"
        );
        assert_ne!(
            base,
            identity(JointCellsLearnersRequest::new(
                data,
                treatments,
                outcome,
                adjustment,
                vec![vec![1, 0], vec![0, 1]],
                config(),
                learners,
            )),
            "orderings"
        );
    }

    #[test]
    fn a_joint_cells_factory_fit_carries_the_calibration_label_of_its_declared_learners() {
        let ctx = ExecutionContext::for_tests(17);
        let label = |identity: &str| {
            Study::compile_joint_cells_learners(request(identity, table(2000, 31)))
                .unwrap()
                .execute(&ctx)
                .unwrap()
                .calibration_label
        };
        let declared = label(IDENTITY);
        assert!(declared.starts_with("learners=caller_factory:"), "{declared}");
        assert_ne!(declared, label("other learners"));
    }
}
