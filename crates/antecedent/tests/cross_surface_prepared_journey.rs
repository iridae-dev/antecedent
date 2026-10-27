//! One prepared study followed through the plan, the report, the artifact and a refresh.
//!
//! Static and horizon, frequentist and Bayesian, on the cross-surface fixture's line
//! `y = 1 + 2t`. The plan view is read before any estimate and keeps its identity through the
//! estimate and a refresh. The report names the same inference the plan states, before and after
//! execution. The artifact of the first result is read back by an independent consumer and is
//! not moved by a later refresh. A result of one data snapshot is not sealed under a handle that
//! holds another, and a refresh the schema refuses leaves the handle as it was.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::report::ReportFormat;
use antecedent::{
    BayesianConfig, Dag, InferenceMode, PreparedStudy, RefuteSuite, Study, StudyBuilder,
    StudyResult, TemporalDag, TemporalEffectQuery, TemporalPolicy,
};
use antecedent_core::{AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, Lag, VariableId};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_test_support::assert_close;
use antecedent_test_support::slice_cases::{self, Gaussian};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

#[derive(Clone, Copy, Debug)]
struct Cell {
    horizon: bool,
    bayesian: bool,
}

const CELLS: [Cell; 4] = [
    Cell { horizon: false, bayesian: false },
    Cell { horizon: false, bayesian: true },
    Cell { horizon: true, bayesian: false },
    Cell { horizon: true, bayesian: true },
];

impl Cell {
    fn label(self) -> String {
        format!(
            "{}-{}",
            if self.horizon { "horizon" } else { "static" },
            if self.bayesian { "bayesian" } else { "frequentist" }
        )
    }

    /// A frequentist fit is exact on the noiseless line; a Bayesian point estimate is a mean of
    /// conjugate draws and is held to the fixture's own band.
    fn tolerance(self) -> f64 {
        match (self.horizon, self.bayesian) {
            (_, true) => 0.1,
            (true, false) => 1e-6,
            (false, false) => 1e-9,
        }
    }
}

fn schema() -> antecedent_core::CausalSchema {
    CausalSchemaBuilder::new()
        .continuous("t")
        .treatment()
        .continuous("y")
        .outcome()
        .build()
        .unwrap()
}

fn builder(cell: Cell, treatment: &[f64], outcome: &[f64], gaussian: &Gaussian) -> StudyBuilder {
    let base = if cell.horizon {
        let series =
            TimeSeriesData::from_f64_columns([("t", treatment), ("y", outcome)], 1).unwrap();
        let treatment_id = VariableId::from_raw(0);
        let outcome_id = VariableId::from_raw(1);
        let mut graph = TemporalDag::empty();
        let lagged =
            antecedent_graph::ensure_lagged(&mut graph, treatment_id, Lag::from_raw(1)).unwrap();
        let now =
            antecedent_graph::ensure_lagged(&mut graph, outcome_id, Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(lagged, now).unwrap();
        let query = TemporalEffectQuery::pulse(treatment_id, outcome_id, 1.0)
            .with_policy(TemporalPolicy::pulse(-1))
            .with_horizon_steps(1);
        Study::series(series).graph(graph).temporal_query(query)
    } else {
        let schema = schema();
        let data =
            TabularData::try_from_schema_f64(schema.clone(), [("t", treatment), ("y", outcome)])
                .unwrap();
        let graph = Dag::from_named_edges(&schema, &[("t", "y")]).unwrap();
        let query =
            AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap());
        Study::tabular(data).graph(graph).query(query)
    };
    if cell.bayesian {
        base.inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate()
                .n_draws(gaussian.posterior_draws)
                .prior_scale(gaussian.prior_scale),
        ))
    } else {
        base
    }
}

fn columns(cell: Cell, gaussian: &Gaussian) -> (&[f64], &[f64]) {
    if cell.horizon {
        (gaussian.horizon_treatment.as_slice(), gaussian.horizon_outcome.as_slice())
    } else {
        (gaussian.treatment.as_slice(), gaussian.outcome.as_slice())
    }
}

fn prepare(cell: Cell, treatment: &[f64], outcome: &[f64], gaussian: &Gaussian) -> PreparedStudy {
    builder(cell, treatment, outcome, gaussian)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx())
        .unwrap()
}

/// The outcome after a data change, and the contrast it carries. A static slope change moves the
/// contrast; a horizon level shift keeps the one-step contrast.
fn moved_outcome(cell: Cell, treatment: &[f64], outcome: &[f64], effect: f64) -> (Vec<f64>, f64) {
    if cell.horizon {
        (outcome.iter().map(|value| value + 0.5).collect(), effect)
    } else {
        (outcome.iter().zip(treatment).map(|(value, level)| value + level).collect(), effect + 1.0)
    }
}

fn ate(result: &StudyResult) -> f64 {
    result.estimate().as_effect().expect("effect").ate
}

fn fact(markdown: &str, key: &str) -> String {
    let prefix = format!("- **{key}:** ");
    markdown
        .lines()
        .find_map(|line| line.strip_prefix(prefix.as_str()))
        .unwrap_or_else(|| panic!("no fact {key:?} in\n{markdown}"))
        .to_owned()
}

fn claim_id(bytes: &[u8]) -> [u8; 32] {
    let consumed = antecedent_io::consume_analysis_result(bytes).unwrap();
    consumed.contract.as_ref().and_then(|c| c.claim.as_ref()).expect("an executed claim").claim_id
}

fn consumed_contrast(bytes: &[u8]) -> f64 {
    antecedent_io::consume_analysis_result(bytes).unwrap().body.estimate.expect("contrast")
}

#[test]
fn the_plan_the_report_and_the_artifact_agree_and_survive_a_refresh() {
    let fixture = slice_cases::load();
    let gaussian = &fixture.gaussian;
    let mut readings = Vec::new();
    for cell in CELLS {
        let label = cell.label();
        let (treatment, outcome) = columns(cell, gaussian);
        let mut prepared = prepare(cell, treatment, outcome, gaussian);

        // The plan is read before any estimate and executes nothing.
        let view = prepared.plan_view().unwrap();
        let program = view.identity();
        let statistical = view.statistical().identity();
        let source = view.scientific().source_digest();
        readings.push((cell, format!("{:?}", view.uncertainty().reading())));
        let before = prepared.report(None, &ctx(), ReportFormat::Markdown).unwrap();
        assert!(before.contains("a contract before execution"), "{label}: {before}");
        assert!(!before.contains("## Claim"), "{label}");

        let result = prepared.estimate_retained(&ctx()).unwrap();
        assert_close(ate(&result), gaussian.effect, cell.tolerance());
        assert_eq!(prepared.plan_view().unwrap().identity(), program, "{label}: estimate");

        let after = prepared.report(Some(&result), &ctx(), ReportFormat::Markdown).unwrap();
        assert!(after.contains("## Claim"), "{label}");
        let want = if cell.bayesian { "bayesian" } else { "frequentist" };
        assert_eq!(fact(&after, "Inference").to_lowercase(), want, "{label}");
        assert_eq!(
            after,
            prepared.report(Some(&result), &ctx(), ReportFormat::Markdown).unwrap(),
            "{label}: reading the report twice changed it"
        );

        // The independent consumer keeps the answer of this snapshot.
        let earlier = prepared.encode_contracted_result(&result, &label, &ctx()).unwrap();
        assert_close(consumed_contrast(&earlier), gaussian.effect, cell.tolerance());
        if !cell.bayesian {
            let again = prepared.estimate_retained(&ctx()).unwrap();
            // The execution receipt records the work actually done, so a re-estimate that
            // reuses a cached product differs in its counters; the claim it makes does not.
            let reencoded = prepared.encode_contracted_result(&again, &label, &ctx()).unwrap();
            assert_eq!(
                claim_id(&reencoded),
                claim_id(&earlier),
                "{label}: a frequentist re-estimate on the same rows changed the claim"
            );
            assert_close(consumed_contrast(&reencoded), consumed_contrast(&earlier), 1e-12);
        }

        // A refresh keeps the statistical program. A lowered operation that binds
        // source rows also changes its identity with those rows.
        let (shifted, truth) = moved_outcome(cell, treatment, outcome, gaussian.effect);
        let refreshed = if cell.horizon {
            let series =
                TimeSeriesData::from_f64_columns([("t", treatment), ("y", shifted.as_slice())], 1)
                    .unwrap();
            prepared.refresh_series(series, &ctx()).unwrap()
        } else {
            let data = TabularData::try_from_schema_f64(
                schema(),
                [("t", treatment), ("y", shifted.as_slice())],
            )
            .unwrap();
            rebind_estimate(&mut prepared, data, &ctx()).unwrap()
        };
        assert_close(ate(&refreshed), truth, 2.0 * cell.tolerance());
        let refreshed_view = prepared.plan_view().unwrap();
        assert_eq!(refreshed_view.statistical().identity(), statistical, "{label}: refresh");
        if source.is_some() {
            assert_ne!(refreshed_view.scientific().source_digest(), source, "{label}: source");
            assert_ne!(refreshed_view.identity(), program, "{label}: source-bound operation");
        } else {
            assert_eq!(refreshed_view.identity(), program, "{label}: refresh");
        }
        let later = prepared.encode_contracted_result(&refreshed, &label, &ctx()).unwrap();
        assert_ne!(later, earlier, "{label}: the refreshed rows are another snapshot");
        assert_close(consumed_contrast(&later), truth, 2.0 * cell.tolerance());
        // The earlier artifact does not follow the refresh.
        assert_close(consumed_contrast(&earlier), gaussian.effect, cell.tolerance());

        // A result of the earlier snapshot is not sealed under the refreshed handle.
        assert!(
            prepared.encode_contracted_result(&result, &label, &ctx()).is_err(),
            "{label}: a result of another data snapshot was sealed"
        );
    }
    // The plan states a posterior reading for the Bayesian cells and another for the rest.
    for (cell, reading) in &readings {
        for (other, other_reading) in &readings {
            if cell.bayesian != other.bayesian {
                assert_ne!(reading, other_reading, "{} against {}", cell.label(), other.label());
            }
        }
    }
}

#[test]
fn a_refresh_the_schema_refuses_leaves_the_prepared_study_as_it_was() {
    let fixture = slice_cases::load();
    let gaussian = &fixture.gaussian;
    let cell = Cell { horizon: false, bayesian: false };
    let (treatment, outcome) = columns(cell, gaussian);
    let mut prepared = prepare(cell, treatment, outcome, gaussian);
    let program = prepared.plan_view().unwrap().identity();
    let first = prepared.estimate_retained(&ctx()).unwrap();
    let before = prepared.encode_contracted_result(&first, "first", &ctx()).unwrap();

    let other = CausalSchemaBuilder::new()
        .continuous("t")
        .treatment()
        .continuous("w")
        .outcome()
        .build()
        .unwrap();
    let foreign =
        TabularData::try_from_schema_f64(other, [("t", treatment), ("w", outcome)]).unwrap();
    assert!(rebind_estimate(&mut prepared, foreign, &ctx()).is_err());

    assert_eq!(prepared.plan_view().unwrap().identity(), program);
    let again = prepared.estimate_retained(&ctx()).unwrap();
    assert_close(ate(&again), ate(&first), 1e-12);
    let reencoded = prepared.encode_contracted_result(&again, "first", &ctx()).unwrap();
    assert_eq!(claim_id(&reencoded), claim_id(&before));
    assert_close(consumed_contrast(&reencoded), consumed_contrast(&before), 1e-12);
}

#[test]
fn a_result_of_another_handle_is_not_sealed_under_this_one() {
    let fixture = slice_cases::load();
    let gaussian = &fixture.gaussian;
    for cell in CELLS {
        let (treatment, outcome) = columns(cell, gaussian);
        let first = prepare(cell, treatment, outcome, gaussian);
        let (shifted, _) = moved_outcome(cell, treatment, outcome, gaussian.effect);
        let second = prepare(cell, treatment, &shifted, gaussian);
        let result = first.estimate_retained(&ctx()).unwrap();
        assert!(first.encode_contracted_result(&result, "own", &ctx()).is_ok(), "{}", cell.label());
        assert!(
            second.encode_contracted_result(&result, "foreign", &ctx()).is_err(),
            "{}: a result of another data snapshot was sealed",
            cell.label()
        );
    }
}

/// Bind `data` on a new branch, materialize and estimate it, then adopt the branch as the handle.
/// A failure leaves the handle unchanged, as the immutable lifecycle does.
fn rebind_estimate(
    prepared: &mut antecedent::PreparedStudy,
    data: antecedent_data::TabularData,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent::StudyResult, antecedent::CausalError> {
    let branch = prepared.rebind_data(data, ctx)?.materialize(ctx)?;
    let result = branch.estimate_retained(ctx)?;
    *prepared = branch;
    Ok(result)
}
