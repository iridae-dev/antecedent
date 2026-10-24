//! Lowered programs of the static average-effect, conditional-effect,
//! interventional-distribution and path-specific cells.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_contract::ClaimKind;
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, CounterfactualQuery, ExecutionContext,
    Intervention, InterventionalDistributionQuery, PathSpecificEffectQuery, TargetForm, Value,
    VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Admg, Cpdag, Dag, DenseNodeId, Pag};

use crate::analysis::StudyBuilder;
use crate::analysis::lowering::registry::REGISTERED;
use crate::planner::PhysicalExecutionPlan;
use crate::planner::lowered::{
    FunctionalKind, LoweredProgram, LoweringDepth, ProblemTarget, UncertaintyReading, WidthSource,
};
use crate::planner::statistical::{ModelRole, Realization, ResampleUnit, ResamplingMethod};
use crate::{BayesianConfig, EstimatorId, IdentifierId, InferenceMode, RefuteSuite, Study};

const FAMILY: &str = "static_effect";

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn d(raw: u32) -> DenseNodeId {
    DenseNodeId::from_raw(raw)
}

fn wave(i: u32, frequency: f64) -> f64 {
    (f64::from(i) * frequency).sin()
}

/// Columns `t, y, z`: `z` confounds a binary `t` and `y = 2 t + z + noise`.
fn confounded_data() -> TabularData {
    let z: Vec<f64> = (0..240_u32).map(|i| wave(i, 0.37)).collect();
    let t: Vec<f64> = (0..240_u32)
        .zip(&z)
        .map(|(i, z)| f64::from(u8::from(z + 0.8 * wave(i, 1.31) > 0.0)))
        .collect();
    let y: Vec<f64> = (0..240_u32)
        .zip(t.iter().zip(&z))
        .map(|(i, (t, z))| 2.0 * t + z + 0.1 * wave(i, 2.17))
        .collect();
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

/// `z -> t`, `z -> y`, `t -> y`.
fn confounded_dag() -> Dag {
    let mut graph = Dag::with_variables(3);
    graph.insert_directed(d(2), d(0)).unwrap();
    graph.insert_directed(d(2), d(1)).unwrap();
    graph.insert_directed(d(0), d(1)).unwrap();
    graph
}

/// `z — t`, `z -> y`, `t -> y`: two completions with different adjustment sets.
fn two_completion_cpdag() -> Cpdag {
    let mut graph = Cpdag::with_variables(3);
    graph.insert_directed(d(2), d(1)).unwrap();
    graph.insert_directed(d(0), d(1)).unwrap();
    graph.insert_undirected(d(2), d(0)).unwrap();
    graph
}

/// Columns `t, y, z, x`; `x` modifies the effect of `t` on `y`.
fn modifier_data() -> TabularData {
    let z: Vec<f64> = (0..240_u32).map(|i| wave(i, 0.37)).collect();
    let x: Vec<f64> = (0..240_u32).map(|i| f64::from(u8::from(i % 2 == 0))).collect();
    let t: Vec<f64> = (0..240_u32).zip(&z).map(|(i, z)| 0.8 * z + wave(i, 1.31)).collect();
    let y: Vec<f64> = (0..240_u32)
        .map(|i| {
            let row = usize::try_from(i).unwrap();
            t[row] + 0.8 * t[row] * x[row] + z[row] + 0.1 * wave(i, 2.17)
        })
        .collect();
    TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
        ("x", x.as_slice()),
    ])
    .unwrap()
}

fn modifier_edges() -> [(u32, u32); 4] {
    [(2, 0), (2, 1), (0, 1), (3, 1)]
}

fn conditional_query() -> CausalQuery {
    CausalQuery::ConditionalEffect(
        ConditionalEffectQuery::try_new(
            AverageEffectQuery::binary_ate(v(0), v(1)).with_effect_modifiers([v(3)]),
        )
        .unwrap(),
    )
}

/// Binary columns from `(levels, count)` cells.
fn expand(names: &[&str], cells: &[(&[f64], usize)]) -> TabularData {
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
    for (levels, count) in cells {
        for (column, level) in columns.iter_mut().zip(levels.iter()) {
            column.extend(std::iter::repeat_n(*level, *count));
        }
    }
    TabularData::from_f64_columns(
        names.iter().zip(&columns).map(|(name, column)| (*name, column.as_slice())),
    )
    .unwrap()
}

/// Binary columns `t, y, z` with every `(z, t)` stratum populated.
fn binary_data() -> TabularData {
    expand(
        &["t", "y", "z"],
        &[
            (&[0.0, 0.0, 0.0], 21),
            (&[0.0, 1.0, 0.0], 9),
            (&[1.0, 0.0, 0.0], 4),
            (&[1.0, 1.0, 0.0], 16),
            (&[0.0, 0.0, 1.0], 12),
            (&[0.0, 1.0, 1.0], 3),
            (&[1.0, 0.0, 1.0], 14),
            (&[1.0, 1.0, 1.0], 21),
        ],
    )
}

fn distribution_query() -> InterventionalDistributionQuery {
    InterventionalDistributionQuery::new(v(1), [Intervention::set(v(0), Value::f64(1.0))])
}

/// Binary columns `t, m, y` of a mediated effect.
fn mediated_data() -> TabularData {
    expand(
        &["t", "m", "y"],
        &[
            (&[0.0, 0.0, 0.0], 40),
            (&[0.0, 0.0, 1.0], 10),
            (&[0.0, 1.0, 0.0], 10),
            (&[0.0, 1.0, 1.0], 40),
            (&[1.0, 0.0, 0.0], 10),
            (&[1.0, 0.0, 1.0], 10),
            (&[1.0, 1.0, 0.0], 10),
            (&[1.0, 1.0, 1.0], 70),
        ],
    )
}

fn path_query() -> PathSpecificEffectQuery {
    PathSpecificEffectQuery::binary(v(0), v(2)).with_path_nodes([v(1)])
}

fn base(data: TabularData) -> StudyBuilder {
    Study::tabular(data).refute(RefuteSuite::None)
}

fn compile(study: &Study) -> PhysicalExecutionPlan {
    study.compile(&ExecutionContext::for_tests(5)).unwrap()
}

fn lowered(plan: &PhysicalExecutionPlan) -> &LoweredProgram {
    plan.lowered().expect("compile attaches a lowered program")
}

/// One study per cell of the family.
#[allow(clippy::too_many_lines, reason = "one fixture per route cell, listed cell by cell")]
fn representatives() -> Vec<(&'static str, Study)> {
    let ate = || AverageEffectQuery::binary_ate(v(0), v(1));
    let modifier_dag = || {
        let mut graph = Dag::with_variables(4);
        for (from, to) in modifier_edges() {
            graph.insert_directed(d(from), d(to)).unwrap();
        }
        graph
    };
    let modifier_pag = {
        let mut graph = Pag::with_variables(4);
        for (from, to) in modifier_edges() {
            graph.insert_directed(d(from), d(to)).unwrap();
        }
        graph
    };
    let ate_pag = {
        let mut graph = Pag::with_variables(3);
        graph.insert_directed(d(2), d(0)).unwrap();
        graph.insert_directed(d(0), d(1)).unwrap();
        graph
    };
    // A disconnected latent pair keeps the graph mixed and leaves the effect
    // identified by adjustment on `z`.
    let latent_admg = {
        let mut graph = Admg::with_variables(5);
        graph.insert_directed(d(2), d(0)).unwrap();
        graph.insert_directed(d(2), d(1)).unwrap();
        graph.insert_directed(d(0), d(1)).unwrap();
        graph.insert_bidirected(d(3), d(4)).unwrap();
        graph
    };
    let latent_data = expand(
        &["t", "y", "z", "u", "v"],
        &[
            (&[0.0, 0.0, 0.0, 0.0, 0.0], 80),
            (&[0.0, 1.0, 0.0, 1.0, 1.0], 20),
            (&[1.0, 0.0, 0.0, 0.0, 0.0], 20),
            (&[1.0, 1.0, 0.0, 1.0, 1.0], 80),
            (&[0.0, 0.0, 1.0, 1.0, 1.0], 60),
            (&[0.0, 1.0, 1.0, 0.0, 0.0], 40),
            (&[1.0, 0.0, 1.0, 1.0, 1.0], 40),
            (&[1.0, 1.0, 1.0, 0.0, 0.0], 60),
        ],
    );
    // `z -> x`, `z -> y`, `x -> y`, `z <-> x`: the conditional effect of `x` on `y`
    // given `z` is identified by IDC.
    let modifier_admg = {
        let mut graph = Admg::with_variables(3);
        graph.insert_directed(d(0), d(1)).unwrap();
        graph.insert_directed(d(0), d(2)).unwrap();
        graph.insert_directed(d(1), d(2)).unwrap();
        graph.insert_bidirected(d(0), d(1)).unwrap();
        graph
    };
    let modifier_admg_data = expand(
        &["z", "x", "y"],
        &[
            (&[0.0, 0.0, 0.0], 30),
            (&[0.0, 0.0, 1.0], 10),
            (&[0.0, 1.0, 0.0], 15),
            (&[0.0, 1.0, 1.0], 25),
            (&[1.0, 0.0, 0.0], 12),
            (&[1.0, 0.0, 1.0], 28),
            (&[1.0, 1.0, 0.0], 8),
            (&[1.0, 1.0, 1.0], 52),
        ],
    );
    let modifier_admg_query = CausalQuery::ConditionalEffect(
        ConditionalEffectQuery::try_new(
            AverageEffectQuery::binary_ate(v(1), v(2)).with_effect_modifiers([v(0)]),
        )
        .unwrap(),
    );
    // `z -> t -> y` with `z <-> y`: adjusting `z` blocks the only back door.
    let distribution_admg = {
        let mut graph = Admg::with_variables(3);
        graph.insert_directed(d(2), d(0)).unwrap();
        graph.insert_directed(d(0), d(1)).unwrap();
        graph.insert_bidirected(d(2), d(1)).unwrap();
        graph
    };
    let mediated_dag = {
        let mut graph = Dag::with_variables(3);
        graph.insert_directed(d(0), d(1)).unwrap();
        graph.insert_directed(d(1), d(2)).unwrap();
        graph
    };
    // `t -> m -> y` with `t <-> y`: every directed path is selected, so the effect is the
    // front-door total effect of the latent-confounded treatment.
    let mediated_admg = {
        let mut graph = Admg::with_variables(3);
        graph.insert_directed(d(0), d(1)).unwrap();
        graph.insert_directed(d(1), d(2)).unwrap();
        graph.insert_bidirected(d(0), d(2)).unwrap();
        graph
    };
    // `t -> m -> y` with both edges compelled: the class has one completion.
    let mediated_cpdag = {
        let mut graph = Cpdag::with_variables(3);
        graph.insert_directed(d(0), d(1)).unwrap();
        graph.insert_directed(d(1), d(2)).unwrap();
        graph
    };
    // `t -> m -> y` with every mark decided: the MAG completion is the graph itself.
    let mediated_pag = {
        let mut graph = Pag::with_variables(3);
        graph.insert_directed(d(0), d(1)).unwrap();
        graph.insert_directed(d(1), d(2)).unwrap();
        graph
    };
    let build = |name: &'static str, builder: StudyBuilder| {
        let study = builder.build().unwrap_or_else(|error| panic!("{name}: {error}"));
        (name, study)
    };
    vec![
        build(
            "StaticAteDag",
            base(confounded_data()).graph(confounded_dag()).query(ate()).bootstrap_replicates(0),
        ),
        build("StaticAtePag", base(confounded_data()).graph(ate_pag).query(ate())),
        build(
            "StaticAteCpdag",
            base(confounded_data()).graph(Cpdag::from_dag(&confounded_dag())).query(ate()),
        ),
        build(
            "StaticAteMixed",
            base(latent_data)
                .graph(latent_admg)
                .query(ate())
                .identifier(IdentifierId::GeneralId)
                .estimator(EstimatorId::FunctionalEffect)
                .bootstrap_replicates(0),
        ),
        build(
            "ConditionalDag",
            base(modifier_data()).graph(modifier_dag()).query(conditional_query()),
        ),
        build(
            "ConditionalCpdag",
            base(modifier_data())
                .graph(Cpdag::from_dag(&modifier_dag()))
                .query(conditional_query()),
        ),
        build(
            "ConditionalPag",
            base(modifier_data()).graph(modifier_pag).query(conditional_query()),
        ),
        build(
            "ConditionalMixed",
            base(modifier_admg_data).graph(modifier_admg).query(modifier_admg_query),
        ),
        build(
            "DistributionDag",
            base(binary_data())
                .graph(confounded_dag())
                .query(CausalQuery::Distribution(distribution_query())),
        ),
        build(
            "DistributionMixed",
            base(binary_data())
                .graph(distribution_admg)
                .query(CausalQuery::Distribution(distribution_query())),
        ),
        build(
            "PathSpecificDag",
            base(mediated_data())
                .graph(mediated_dag)
                .query(CausalQuery::PathSpecific(path_query())),
        ),
        build(
            "PathSpecificCpdag",
            base(mediated_data())
                .graph(mediated_cpdag)
                .query(CausalQuery::PathSpecific(path_query())),
        ),
        build(
            "PathSpecificPag",
            base(mediated_data())
                .graph(mediated_pag)
                .query(CausalQuery::PathSpecific(path_query())),
        ),
        // `t -> m -> y` with `t <-> y`: ID* identifies the effect on the treated by the front door.
        build(
            "CounterfactualClass",
            base(mediated_data())
                .graph(mediated_admg.clone())
                .query(CausalQuery::Counterfactual(
                    CounterfactualQuery::new(v(2), [Intervention::set(v(0), Value::f64(1.0))])
                        .with_control_level(0.0),
                ))
                .bootstrap_replicates(0),
        ),
        build(
            "PathSpecificMixed",
            base(mediated_data())
                .graph(mediated_admg)
                .query(CausalQuery::PathSpecific(path_query())),
        ),
    ]
}

#[test]
fn every_cell_of_the_family_lowers_completely() {
    let family: BTreeSet<&str> = REGISTERED
        .iter()
        .filter(|(_, family)| *family == FAMILY)
        .map(|(cell, _)| cell.as_str())
        .collect();
    let mut covered = BTreeSet::new();
    for (name, study) in representatives() {
        let plan = study.compile(&ExecutionContext::for_tests(5)).unwrap_or_else(|error| {
            panic!("{name}: {error}");
        });
        let program = lowered(&plan);
        crate::planner::plan_view::checks::assert_view_agrees(name, &plan);
        assert_eq!(program.route(), name, "the representative of {name} compiles as another cell");
        assert_eq!(program.depth(), LoweringDepth::Complete, "{name} lowers completely");
        assert!(program.statistical().estimator().is_some(), "{name} names no estimator");
        let again = compile(&study);
        assert_eq!(lowered(&again).identity(), program.identity(), "{name} identity is unstable");
        covered.insert(name);
    }
    assert_eq!(covered, family, "every cell of the family has a representative");
}

#[test]
fn average_effect_binds_its_contrast_estimator_models_and_resampling() {
    let ctx = ExecutionContext::for_tests(7);
    let study = |estimator: EstimatorId, replicates: u32| {
        base(confounded_data())
            .graph(confounded_dag())
            .query(AverageEffectQuery::binary_ate(v(0), v(1)))
            .estimator(estimator)
            .bootstrap_replicates(replicates)
            .build()
            .unwrap()
    };
    let weighting = study(EstimatorId::PropensityWeighting, 20);
    let plan = compile(&weighting);
    let program = lowered(&plan);
    assert_eq!(program.route(), "StaticAteDag");
    assert_eq!(program.depth(), LoweringDepth::Complete);
    let ProblemTarget::Target(target) = program.scientific().target() else {
        panic!("an average effect states a world-indexed target");
    };
    let TargetForm::Contrast(contrast) = target.form() else {
        panic!("an average effect is the contrast of two worlds");
    };
    assert_eq!(contrast.worlds().worlds().len(), 2);
    assert!(program.identification_retained(), "compile retains the identification");

    let statistical = program.statistical();
    assert_eq!(statistical.estimator(), Some(EstimatorId::PropensityWeighting));
    assert!(matches!(statistical.realization(), Realization::Frequentist(_)));
    let resampling = statistical.resampling().expect("the study bootstraps");
    assert_eq!(resampling.replicates(), 20);
    assert_eq!(resampling.method(), ResamplingMethod::Bootstrap);
    assert_eq!(resampling.unit(), ResampleUnit::Row);
    assert!(statistical.posterior().is_none());
    let propensity = statistical
        .models()
        .iter()
        .find(|model| model.role() == ModelRole::Propensity)
        .expect("weighting fits a propensity model");
    assert_eq!(propensity.output(), Some(v(0)));
    assert!(propensity.inputs().contains(&v(2)), "the propensity reads the adjustment set");
    assert_eq!(program.claim().kind(), Some(ClaimKind::Point));
    assert_eq!(program.claim().uncertainty(), UncertaintyReading::Sampling);
    assert_eq!(
        program.physical().operators().len(),
        5,
        "read, fit and intervene, then the projection and the scan"
    );

    let fewer = compile(&study(EstimatorId::PropensityWeighting, 10));
    assert_ne!(lowered(&fewer).identity(), program.identity(), "replicates are in the identity");
    let linear = compile(&study(EstimatorId::LinearAdjustmentAte, 20));
    assert_ne!(lowered(&linear).identity(), program.identity(), "the estimator is in the identity");
    let shifted = base(confounded_data())
        .graph(confounded_dag())
        .query(AverageEffectQuery::with_levels(v(0), v(1), 0.0, 2.0))
        .estimator(EstimatorId::PropensityWeighting)
        .bootstrap_replicates(20)
        .build()
        .unwrap();
    assert_ne!(
        lowered(&compile(&shifted)).identity(),
        program.identity(),
        "the target is in the identity"
    );

    let prepared = weighting.prepare(&ctx).unwrap();
    assert_eq!(prepared.plan().lowered_program_identity(), Some(program.identity()));
    let estimated = prepared.estimate_retained(&ctx).unwrap();
    assert_eq!(estimated.lowered_program_identity(), Some(program.identity()));
}

#[test]
fn linear_adjustment_quotes_its_design_and_keeps_the_checked_target() {
    let study = base(confounded_data())
        .graph(confounded_dag())
        .query(AverageEffectQuery::binary_ate(v(0), v(1)))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let plan = compile(&study);
    let program = lowered(&plan);
    let checked = plan.compiled_linear_program().expect("the default estimator is retained");
    let ProblemTarget::Target(target) = program.scientific().target() else {
        panic!("the linear cell lowers its checked target");
    };
    assert_eq!(target.identity(), checked.target().identity());
    let design = program.physical().design();
    assert_eq!(design.source(), WidthSource::Quoted);
    assert_eq!(design.columns(), Some(3), "intercept, treatment and the adjustment column");
    assert!(program.statistical().resampling().is_none(), "no replicates were requested");
    let outcome = &program.statistical().models()[0];
    assert_eq!(outcome.role(), ModelRole::Outcome);
    assert_eq!(outcome.output(), Some(v(1)));
}

#[test]
fn bayesian_average_effect_lowers_a_posterior_program() {
    let ctx = ExecutionContext::for_tests(9);
    let study = base(confounded_data())
        .graph(confounded_dag())
        .query(AverageEffectQuery::binary_ate(v(0), v(1)))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let plan = compile(&study);
    let program = lowered(&plan);
    assert_eq!(program.depth(), LoweringDepth::Complete);
    let statistical = program.statistical();
    assert_eq!(statistical.estimator(), Some(EstimatorId::BayesianGcomp));
    assert!(matches!(statistical.realization(), Realization::Bayesian(_)));
    assert_eq!(statistical.posterior().expect("a posterior program").draws(), 64);
    assert!(statistical.resampling().is_none());
    assert_eq!(program.claim().uncertainty(), UncertaintyReading::Posterior);

    let frequentist = base(confounded_data())
        .graph(confounded_dag())
        .query(AverageEffectQuery::binary_ate(v(0), v(1)))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    assert_ne!(
        lowered(&compile(&frequentist)).identity(),
        program.identity(),
        "the inference mode is in the identity"
    );
    let prepared = study.prepare(&ctx).unwrap();
    assert_eq!(prepared.plan().lowered_program_identity(), Some(program.identity()));
    let estimated = prepared.estimate_retained(&ctx).unwrap();
    assert_eq!(estimated.lowered_program_identity(), Some(program.identity()));
}

#[test]
fn partially_identified_class_effect_lowers_an_identified_set() {
    let ctx = ExecutionContext::for_tests(13);
    let study = base(confounded_data())
        .graph(two_completion_cpdag())
        .query(AverageEffectQuery::with_levels(v(0), v(1), 0.0, 1.0))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let plan = compile(&study);
    let program = lowered(&plan);
    assert_eq!(program.route(), "StaticAteCpdag");
    assert_eq!(program.depth(), LoweringDepth::Complete);
    assert!(program.identification_retained());
    assert!(matches!(program.statistical().realization(), Realization::Envelope(_)));
    assert_eq!(program.claim().kind(), Some(ClaimKind::Bounds));
    assert_eq!(program.claim().uncertainty(), UncertaintyReading::Envelope);
    let prepared = study.prepare(&ctx).unwrap();
    let estimated = prepared.estimate_retained(&ctx).unwrap();
    assert_eq!(estimated.lowered_program_identity(), Some(program.identity()));
}

#[test]
fn conditional_effect_conditions_its_target_on_the_modifiers() {
    let study = base(modifier_data())
        .graph({
            let mut graph = Dag::with_variables(4);
            for (from, to) in modifier_edges() {
                graph.insert_directed(d(from), d(to)).unwrap();
            }
            graph
        })
        .query(conditional_query())
        .build()
        .unwrap();
    let plan = compile(&study);
    let program = lowered(&plan);
    let ProblemTarget::Target(target) = program.scientific().target() else {
        panic!("a conditional effect states a world-indexed target");
    };
    let TargetForm::Contrast(contrast) = target.form() else {
        panic!("a conditional effect is a conditioned contrast");
    };
    for world in contrast.worlds().worlds() {
        assert_eq!(world.binding().conditioning().variables(), &[v(3)]);
    }
    let outcome = program
        .statistical()
        .models()
        .iter()
        .find(|model| model.role() == ModelRole::Outcome)
        .expect("a conditional effect fits an outcome model");
    assert!(outcome.inputs().contains(&v(3)), "the outcome model reads the modifier");
}

#[test]
fn distribution_functionals_of_different_queries_lower_to_different_programs() {
    let at = |level: f64| {
        let action = Intervention::set(v(0), Value::f64(level));
        let query = InterventionalDistributionQuery::new(v(1), [action]);
        let study = base(binary_data())
            .graph(confounded_dag())
            .query(CausalQuery::Distribution(query))
            .build()
            .unwrap();
        let plan = compile(&study);
        let program = lowered(&plan);
        let ProblemTarget::QueryFunctional { query, .. } = program.scientific().target() else {
            panic!("an interventional distribution lowers to its typed functional");
        };
        (*query, program.identity())
    };
    let (one, one_identity) = at(1.0);
    let (zero, zero_identity) = at(0.0);
    assert_ne!(one, zero, "the query digest reads the intervention level");
    assert_ne!(one_identity, zero_identity, "two queries of one kind are two programs");
    assert_eq!(at(1.0), (one, one_identity), "the digest is canonical across compiles");
}

#[test]
fn a_weighting_route_quotes_the_width_its_nuisance_models_share() {
    let weighting = base(confounded_data())
        .graph(confounded_dag())
        .query(AverageEffectQuery::binary_ate(v(0), v(1)))
        .estimator(EstimatorId::PropensityWeighting)
        .bootstrap_replicates(10)
        .build()
        .unwrap();
    let plan = compile(&weighting);
    let design = lowered(&plan).physical().design();
    assert_eq!(design.source(), WidthSource::Quoted, "the adjustment set is one fixed set");
    let columns = design.columns().expect("a quoted design has columns");
    assert!(columns >= 3, "an intercept, the treatment and the adjustment: {columns}");
    assert_eq!(design.rows(), Some(plan.logical.row_count_hint));
}

#[test]
fn distribution_and_path_specific_cells_state_their_own_targets() {
    let ctx = ExecutionContext::for_tests(17);
    let distribution = base(binary_data())
        .graph(confounded_dag())
        .query(CausalQuery::Distribution(distribution_query()))
        .build()
        .unwrap();
    let plan = compile(&distribution);
    let program = lowered(&plan);
    assert!(matches!(
        program.scientific().target(),
        ProblemTarget::QueryFunctional {
            kind: FunctionalKind::InterventionalDistribution,
            outcome: Some(outcome),
            ..
        } if *outcome == v(1)
    ));
    assert_eq!(program.statistical().estimator(), Some(EstimatorId::FunctionalDistribution));
    assert!(matches!(program.physical().design().source(), WidthSource::NotDesignBased(_)));
    let prepared = distribution.prepare(&ctx).unwrap();
    let estimated = prepared.estimate_retained(&ctx).unwrap();
    assert_eq!(estimated.lowered_program_identity(), Some(program.identity()));

    let mediated_dag = {
        let mut graph = Dag::with_variables(3);
        graph.insert_directed(d(0), d(1)).unwrap();
        graph.insert_directed(d(1), d(2)).unwrap();
        graph
    };
    let path = base(mediated_data())
        .graph(mediated_dag)
        .query(CausalQuery::PathSpecific(path_query()))
        .build()
        .unwrap();
    let plan = compile(&path);
    let program = lowered(&plan);
    let ProblemTarget::Target(target) = program.scientific().target() else {
        panic!("a path-specific effect through declared nodes is a mediation target");
    };
    let TargetForm::Mediation(mediation) = target.form() else {
        panic!("a path-specific effect through declared nodes is a mediation target");
    };
    assert_eq!(mediation.arms().mediators(), &[v(1)]);
    let prepared = path.prepare(&ctx).unwrap();
    let estimated = prepared.estimate_retained(&ctx).unwrap();
    assert_eq!(estimated.lowered_program_identity(), Some(program.identity()));
}
