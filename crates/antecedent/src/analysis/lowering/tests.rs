//! Registry and guard checks for route lowering.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_core::{AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, SemanticDigest};
use antecedent_data::TabularData;

use super::registry::REGISTERED;
use super::{GraphPosteriorCell, RouteCell};
use crate::planner::lowered::LoweringDepth;
use crate::{Dag, RefuteSuite, Study};

const FAMILIES: [&str; 6] =
    ["static_effect", "static_response", "structural", "temporal", "design_based", "discovery"];

/// Registered cells no licensed support-matrix coordinate executes: the graphless
/// design axes, the panel and multi-environment modalities, the latent-confounded
/// temporal PAG mediation every compile refuses, and the structure-discovery,
/// attribution and simulation cells that execute unlicensed. Each still lowers
/// completely through its family; the licensed-route record simply has no row for it.
const OUTSIDE_LICENSED_ROUTES: [&str; 21] = [
    "DiscoverStatic",
    "DiscoverTemporal",
    "GcmAttributeDag",
    "GcmSimulateDag",
    "TemporalMediationPag",
    "MultiEnvTemporalEffect",
    "PanelTemporalEffectDag",
    "PanelTemporalEffectClass",
    "PanelTemporalResponseDag",
    "PanelTemporalResponseClass",
    "RandomizedEffect",
    "PolicyValue",
    "PanelDid",
    "SyntheticControl",
    "ContinuousDoseResponse",
    "LocalPolynomialRatio",
    "Survival",
    "LongitudinalRegime",
    "SequentialContrast",
    "HistoryMatch",
    "TemporalDesign",
];

/// Licensed routes the support matrix executes.
const LICENSED_ROUTE_COUNT: usize = 723;

/// The `lowering` value of every `[[route]]` in the licensed-route record, in order.
fn licensed_route_lowerings() -> Vec<Option<String>> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/licensed_routes.toml");
    let text = std::fs::read_to_string(path).expect("parity/licensed_routes.toml");
    let mut routes: Vec<Option<String>> = Vec::new();
    for line in text.lines().map(str::trim) {
        if line == "[[route]]" {
            routes.push(None);
        } else if let Some(value) = line.strip_prefix("lowering = ") {
            let route = routes.last_mut().expect("a lowering value belongs to a [[route]]");
            *route = Some(value.trim_matches('"').to_owned());
        }
    }
    routes
}

#[test]
fn every_registered_cell_lowers_completely_and_the_licensed_routes_name_registered_cells() {
    // One depth: a partial lowering cannot be added without failing this match.
    let complete = |depth: LoweringDepth| match depth {
        LoweringDepth::Complete => true,
    };
    assert!(complete(LoweringDepth::Complete));
    let mut registered: BTreeSet<&str> = REGISTERED.iter().map(|(cell, _)| cell.as_str()).collect();
    for cell in GraphPosteriorCell::ALL {
        registered.insert(RouteCell::GraphPosterior(cell).as_str());
    }
    let routes = licensed_route_lowerings();
    assert_eq!(routes.len(), LICENSED_ROUTE_COUNT, "licensed route inventory drifted");
    let mut named = BTreeSet::new();
    for (index, route) in routes.iter().enumerate() {
        let lowering = route
            .as_deref()
            .unwrap_or_else(|| panic!("licensed route #{} records no lowering", index + 1));
        assert!(registered.contains(lowering), "licensed route names unregistered {lowering}");
        named.insert(lowering);
    }
    let outside: BTreeSet<&str> = OUTSIDE_LICENSED_ROUTES.into_iter().collect();
    assert!(outside.is_subset(&registered), "an excluded cell is not registered");
    assert!(named.is_disjoint(&outside), "a licensed route names an excluded cell");
    let covered: BTreeSet<&str> = named.union(&outside).copied().collect();
    assert_eq!(covered, registered, "every registered cell is licensed or explicitly excluded");
}

#[test]
fn every_route_cell_is_registered_once_to_a_family() {
    let mut names = BTreeSet::new();
    for (cell, family) in REGISTERED {
        assert!(FAMILIES.contains(family), "{} is registered to {family}", cell.as_str());
        assert!(names.insert(cell.as_str()), "{} is registered twice", cell.as_str());
    }
    assert_eq!(names.len(), 50, "route cell inventory drifted");
    for cell in GraphPosteriorCell::ALL {
        assert!(names.insert(RouteCell::GraphPosterior(cell).as_str()), "{}", cell.as_str());
    }
}

fn linear_study() -> Study {
    linear_study_with([1.0, 3.1, 0.9, 2.9, 1.0, 3.0])
}

/// `linear_study` with the outcome values `outcome` on the same schema.
fn linear_study_with(outcome: [f64; 6]) -> Study {
    let schema = CausalSchemaBuilder::new()
        .continuous("t")
        .treatment()
        .continuous("y")
        .outcome()
        .build()
        .unwrap();
    let treatment = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let columns: [(&str, &[f64]); 2] = [("t", &treatment), ("y", &outcome)];
    let data = TabularData::try_from_schema_f64(schema.clone(), columns).unwrap();
    let graph = Dag::from_named_edges(&schema, &[("t", "y")]).unwrap();
    let query =
        AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap());
    Study::tabular(data)
        .graph(graph)
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

#[test]
fn compile_attaches_a_lowered_program_and_execution_stamps_it() {
    let ctx = ExecutionContext::for_tests(3);
    let study = linear_study();
    let plan = study.compile(&ctx).unwrap();
    let lowered = plan.lowered().expect("compile attaches a lowered program");
    assert_eq!(lowered.route(), "StaticAteDag");
    assert_eq!(lowered.depth(), LoweringDepth::Complete);
    let result = study.execute(&plan, &ctx).unwrap();
    assert_eq!(result.lowered_program_identity(), Some(lowered.identity()));
    let prepared = study.prepare(&ctx).unwrap();
    let estimated = prepared.estimate_retained(&ctx).unwrap();
    assert_eq!(estimated.lowered_program_identity(), prepared.plan().lowered_program_identity());
}

#[test]
fn dispatch_refuses_a_plan_without_a_lowered_program() {
    let ctx = ExecutionContext::for_tests(3);
    let study = linear_study();
    let mut plan = study.compile(&ctx).unwrap();
    plan.lowered = None;
    let refusal = study.execute(&plan, &ctx).unwrap_err();
    assert!(refusal.to_string().contains("route_not_lowered"), "{refusal}");
}

#[test]
fn compile_and_run_refuse_inside_a_lowered_dispatch() {
    let ctx =
        ExecutionContext::for_tests(3).enter_lowered_dispatch(SemanticDigest::from_bytes([7; 32]));
    let study = linear_study();
    let compiled = study.compile(&ctx).unwrap_err();
    assert!(compiled.to_string().contains("public_builder_reentry"), "{compiled}");
    assert!(study.run(&ctx).is_err(), "one-shot run inside a lowered dispatch must refuse");
}

#[test]
fn prepare_installs_the_identification_compile_retained() {
    use antecedent_core::execution_attempt::{Operation, observe_execution};
    let ctx = ExecutionContext::for_tests(3);
    let study = linear_study();
    let observed = observe_execution(|| study.prepare(&ctx));
    assert_eq!(
        observed.report.counts(Operation::Identification).attempted,
        1,
        "compile identifies once; preparation installs the retained products"
    );
    let prepared = observed.result.unwrap();
    let lowered = prepared.plan().lowered().unwrap();
    assert!(lowered.identification_retained());
    let program = prepared.plan().compiled_linear_program().unwrap();
    let crate::planner::lowered::ProblemTarget::Target(target) = lowered.scientific().target()
    else {
        panic!("the linear cell lowers its checked target");
    };
    assert_eq!(target.identity(), program.target().identity());
    assert_eq!(lowered.physical().operators().len(), program.physical_operators().len());
}

#[test]
fn quoted_design_width_counts_contrast_columns_the_planner_sizing_misses() {
    use antecedent_core::physical_choice::PhysicalAlternative;
    use antecedent_core::{CategoryDomainId, MeasurementSpec, SmallRoleSet, ValueType};
    use antecedent_data::{
        CategoricalColumn, CategoryCode, CategoryDomain, CategoryLevel, Contrast, Float64Column,
        OwnedColumn, OwnedColumnarStorage, UnknownCategoryPolicy, ValidityBitmap,
    };
    use std::sync::Arc;
    let mut builder =
        CausalSchemaBuilder::new().continuous("t").treatment().continuous("y").outcome();
    builder
        .add_variable(
            "z",
            ValueType::Categorical,
            SmallRoleSet::empty(),
            None,
            Some(CategoryDomainId::from_raw(2)),
            MeasurementSpec::default(),
        )
        .unwrap();
    let schema = builder.build().unwrap();
    let t = schema.id_of("t").unwrap();
    let y = schema.id_of("y").unwrap();
    let z = schema.id_of("z").unwrap();
    let domain = Arc::new(
        CategoryDomain::try_new(
            CategoryDomainId::from_raw(2),
            (0..4)
                .map(|i| CategoryLevel { label: Arc::from(format!("level{i}")) })
                .collect::<Vec<_>>(),
            false,
            Some(CategoryCode::from_raw(0)),
            UnknownCategoryPolicy::Fail,
        )
        .unwrap(),
    );
    let treatment: Vec<_> = (0..64).map(|i| f64::from(i % 2)).collect();
    let category: Vec<_> = (0..64).map(|i| CategoryCode::from_raw((i / 2) % 4)).collect();
    let outcome: Vec<_> = (0..64)
        .map(|i| 2.0 * treatment[i] + f64::from(u32::try_from((i / 2) % 4).unwrap()))
        .collect();
    let columns = vec![
        OwnedColumn::Float64(
            Float64Column::new(t, Arc::<[f64]>::from(treatment), ValidityBitmap::all_valid(64))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(y, Arc::<[f64]>::from(outcome), ValidityBitmap::all_valid(64))
                .unwrap(),
        ),
        OwnedColumn::Categorical(
            CategoricalColumn::try_new(z, category, ValidityBitmap::all_valid(64), domain).unwrap(),
        ),
    ];
    let data = TabularData::new(
        OwnedColumnarStorage::try_new(schema.clone(), columns, None, None).unwrap(),
    );
    let graph = Dag::from_named_edges(&schema, &[("t", "y"), ("z", "t"), ("z", "y")]).unwrap();
    let fitter = antecedent_estimate::LinearAdjustmentAte::new()
        .with_bootstrap_replicates(0)
        .with_contrast(z, Contrast::Treatment { reference: CategoryCode::from_raw(0) });
    let ctx = ExecutionContext::for_tests(1);
    let prepared = Study::tabular(data)
        .graph(graph)
        .query(AverageEffectQuery::binary_ate(t, y))
        .estimator(fitter)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let plan = prepared.plan();
    let design = plan.lowered().unwrap().physical().design();
    let planner_sizing = plan.logical.record.query_variables.len() as u64;
    assert_eq!(planner_sizing, 2, "the planner's sizing counts the query's variables");
    assert_eq!(design.columns(), Some(5), "intercept, treatment and three contrast columns");
    assert_ne!(design.columns(), Some(planner_sizing));
    assert_eq!(design.source(), crate::planner::lowered::WidthSource::Quoted);
    assert!(!design.direct());
    assert_eq!(plan.choice.selected(), PhysicalAlternative::MaterializedDesign);
    let built = prepared.execution().checked_linear().unwrap().problem().design.ncols;
    assert_eq!(design.columns(), Some(built as u64), "the quoted width is the built width");
}

#[test]
fn requote_replans_bytes_and_materialization_under_the_memory_limit() {
    use crate::planner::lowered::DesignShape;
    use antecedent_contract::BufferMaterialization;
    let ctx = ExecutionContext::for_tests(3);
    let mut plan = linear_study().compile(&ctx).unwrap();
    let direct = DesignShape::quoted(10, 3, true);
    plan.requote(direct, &ctx).unwrap();
    assert_eq!(plan.logical.design_columns_hint, 3);
    assert_eq!(plan.record.estimated_copy_bytes, Some(10 * 3 * 8));
    assert_eq!(plan.record.materializations[0].1, BufferMaterialization::Borrowed);
    let materialized = direct.materialized();
    plan.requote(materialized, &ctx).unwrap();
    assert_eq!(plan.record.estimated_copy_bytes, Some(2 * 10 * 3 * 8));
    assert_eq!(plan.record.materializations[0].1, BufferMaterialization::CopiedContiguous);
    let mut bounded = ctx.clone();
    bounded.set_memory_limit(Some(1));
    assert!(matches!(plan.requote(direct, &bounded), Err(crate::CausalError::Resource { .. })));
}

/// The accepted graphs of every class over the two variables of `linear_study`.
/// Each pair of graphs differs, so each must key a different retained product.
fn graphs_of_every_class() -> Vec<(&'static str, crate::AcceptedGraph)> {
    use crate::{AcceptedGraph, TemporalDag};
    use antecedent_core::{Lag, VariableId};
    use antecedent_graph::{
        Admg, Cpdag, DenseNodeId, Pag, TemporalAdmg, TemporalCpdag, TemporalPag, ensure_lagged,
    };
    let (a, b) = (DenseNodeId::from_raw(0), DenseNodeId::from_raw(1));
    let (t, y) = (VariableId::from_raw(0), VariableId::from_raw(1));
    let mut directed = Dag::with_variables(2);
    directed.insert_directed(a, b).unwrap();
    let mut reversed = Dag::with_variables(2);
    reversed.insert_directed(b, a).unwrap();
    let mut admg = Admg::with_variables(2);
    admg.insert_directed(a, b).unwrap();
    let mut confounded = admg.clone();
    confounded.insert_bidirected(a, b).unwrap();
    let mut circle_arrow = Pag::with_variables(2);
    circle_arrow.insert_circle_arrow(a, b).unwrap();
    let mut circle_circle = Pag::with_variables(2);
    circle_circle.insert_circle_circle(a, b).unwrap();
    let mut lagged = TemporalDag::empty();
    let past = ensure_lagged(&mut lagged, t, Lag::from_raw(1)).unwrap();
    let now = ensure_lagged(&mut lagged, y, Lag::CONTEMPORANEOUS).unwrap();
    lagged.insert_directed(past, now).unwrap();
    let mut temporal_pag = TemporalPag::empty();
    let past = temporal_pag.add_lagged(t, Lag::from_raw(1)).unwrap();
    let now = temporal_pag.add_lagged(y, Lag::CONTEMPORANEOUS).unwrap();
    temporal_pag.insert_circle_arrow(past, now).unwrap();
    vec![
        ("reversed dag", AcceptedGraph::dag(reversed)),
        ("empty dag", AcceptedGraph::dag(Dag::with_variables(2))),
        ("admg", AcceptedGraph::admg(admg)),
        ("confounded admg", AcceptedGraph::admg(confounded.clone())),
        ("cpdag", AcceptedGraph::cpdag(Cpdag::from_dag(&directed)).unwrap()),
        ("empty cpdag", AcceptedGraph::cpdag(Cpdag::with_variables(2)).unwrap()),
        ("pag", AcceptedGraph::pag(circle_arrow)),
        ("circle pag", AcceptedGraph::pag(circle_circle)),
        ("temporal dag", AcceptedGraph::temporal_dag(lagged.clone())),
        (
            "temporal cpdag",
            AcceptedGraph::temporal_cpdag(TemporalCpdag::from_temporal_dag(&lagged)).unwrap(),
        ),
        ("temporal pag", AcceptedGraph::temporal_pag(temporal_pag)),
        (
            "temporal admg",
            AcceptedGraph::temporal_admg(
                TemporalAdmg::from_static_contemporaneous(&confounded).unwrap(),
            ),
        ),
    ]
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "a linear test script that changes every identification input in turn"
)]
fn a_retained_identification_is_reused_only_by_a_study_with_the_same_identification_inputs() {
    use std::sync::Arc;

    use antecedent_core::{CausalQuery, VariableId};
    use antecedent_graph::{Admg, SelectionDiagram, TieredBackground, WithinTier};

    use crate::analysis::builder::{RdConfig, TransportTrialSpec};
    use crate::{BayesianConfig, ClassPrior, EstimatorId, IdentifierId, InferenceMode};

    fn config(study: &mut Study) -> &mut crate::analysis::execute::StudyConfig {
        Arc::make_mut(&mut study.config)
    }

    let ctx = ExecutionContext::for_tests(5);
    let study = linear_study();
    let plan = study.compile(&ctx).unwrap();
    assert!(super::plan_retained_static(&study, &plan).is_some(), "the study reuses its own");
    let key = super::retention_key(&study).expect("every input of the study encodes");

    // New values on the same schema change no input identification reads.
    let refreshed = linear_study_with([1.5, 2.0, 0.5, 3.5, 1.0, 2.5]);
    assert_eq!(super::retention_key(&refreshed), Some(key));
    assert!(super::plan_retained_static(&refreshed, &plan).is_some());

    let (t, y) = (VariableId::from_raw(0), VariableId::from_raw(1));
    let edited = |edit: &dyn Fn(&mut Study)| {
        let mut changed = study.clone();
        edit(&mut changed);
        changed
    };
    let mut changed: Vec<(&str, Study)> = vec![
        (
            "effect modifiers",
            edited(&|study| {
                if let CausalQuery::AverageEffect(query) = &mut study.query {
                    query.effect_modifiers = vec![y].into();
                }
            }),
        ),
        (
            "tier background",
            edited(&|study| {
                study.tiered =
                    Some(TieredBackground::new([vec![t], vec![y]], WithinTier::Unknown).unwrap());
            }),
        ),
        ("completion cap", edited(&|study| study.max_completions = Some(3))),
        (
            "class prior",
            edited(&|study| study.class_prior = Some(ClassPrior::from_ordered(vec![1.0]).unwrap())),
        ),
        (
            "selection diagram",
            edited(&|study| {
                let diagram = SelectionDiagram::try_new(Admg::with_variables(2), vec![y]).unwrap();
                config(study).selection_diagram = Some(diagram);
            }),
        ),
        (
            "transport trial",
            edited(&|study| {
                config(study).transport_trial = Some(TransportTrialSpec {
                    trial: t,
                    selection_probability: y,
                    treatment_probability: t,
                });
            }),
        ),
        (
            "discontinuity design",
            edited(&|study| {
                config(study).rd = Some(RdConfig {
                    running_variable: t,
                    cutoff: 0.5,
                    bandwidth: 1.0,
                    se_kind: antecedent_estimate::AnalyticSeKind::Hc1,
                });
            }),
        ),
        ("identifier", edited(&|study| study.identifier = Some(IdentifierId::GeneralId))),
        ("estimator", edited(&|study| study.estimator = Some(EstimatorId::Aipw))),
        (
            "inference",
            edited(&|study| {
                study.inference = InferenceMode::Bayesian(BayesianConfig::conjugate());
            }),
        ),
        (
            "schema",
            edited(&|study| {
                let schema = CausalSchemaBuilder::new()
                    .continuous("t")
                    .treatment()
                    .continuous("y")
                    .outcome()
                    .continuous("z")
                    .context()
                    .build()
                    .unwrap();
                let column = [0.0; 6];
                let columns: [(&str, &[f64]); 3] = [("t", &column), ("y", &column), ("z", &column)];
                let data = TabularData::try_from_schema_f64(schema, columns).unwrap();
                study.data = crate::analysis::builder::DataInput::Tabular(data);
            }),
        ),
    ];
    for (class, graph) in graphs_of_every_class() {
        changed.push((class, edited(&|study| study.graph = Arc::new(graph.clone()))));
    }
    let mut keys = BTreeSet::from([key]);
    for (input, changed) in &changed {
        let changed_key = super::retention_key(changed)
            .unwrap_or_else(|| panic!("{input}: every input has a canonical encoding"));
        assert!(keys.insert(changed_key), "{input}: changing it must change the retention key");
        assert!(
            super::plan_retained_static(changed, &plan).is_none(),
            "{input}: a changed study must not reuse the retained identification"
        );
    }
}

#[test]
fn the_query_digest_ignores_the_order_of_unordered_inputs_and_tracks_every_field() {
    use std::sync::Arc;

    use antecedent_core::{
        AnomalyAttributionQuery, CausalQuery, ContinuousDomain, DerivativeWeighting, GridSpec,
        Intervention, InterventionalDistributionQuery, ObservationAssumption, ObservationSpec,
        RandomizedEffectQuery, ResponseFunctional, ResponseQuery, Value, VariableId,
    };

    let v = VariableId::from_raw;
    let digest = |query: CausalQuery| super::query_digest(&query);
    let average = |modifiers: Vec<VariableId>| {
        let mut query = AverageEffectQuery::binary_ate(v(0), v(1));
        query.effect_modifiers = modifiers.into();
        CausalQuery::AverageEffect(query)
    };
    assert_eq!(digest(average(vec![v(2), v(3)])), digest(average(vec![v(3), v(2)])));
    assert_ne!(digest(average(vec![v(2), v(3)])), digest(average(vec![v(2)])));

    let distribution = |conditioning: Vec<VariableId>| {
        CausalQuery::Distribution(
            InterventionalDistributionQuery::new(
                v(1),
                vec![Intervention::set(v(0), Value::f64(1.0))],
            )
            .with_conditioning(conditioning),
        )
    };
    assert_eq!(digest(distribution(vec![v(2), v(3)])), digest(distribution(vec![v(3), v(2)])));

    let curve = |grid: Vec<f64>| ResponseFunctional::MeanCurve {
        outcome: v(1),
        treatment: ContinuousDomain::new(v(0), GridSpec::Values(grid.into())),
    };
    let response =
        |functional: ResponseFunctional| CausalQuery::Response(ResponseQuery::new(functional));
    let observed = |assumptions: Vec<ObservationAssumption>| {
        CausalQuery::Response(ResponseQuery::new(curve(vec![0.0, 1.0])).with_observation(
            ObservationSpec::RightCensored {
                latent: v(1),
                observed: v(2),
                censoring: v(3),
                event: v(4),
            },
            assumptions,
        ))
    };
    let derivative = |lower: f64| {
        response(ResponseFunctional::AverageDerivative {
            outcome: v(1),
            treatment: v(0),
            weighting: DerivativeWeighting::Uniform { lower, upper: 1.0 },
        })
    };
    let independent = ObservationAssumption::IndependentGiven(vec![v(5), v(6)].into());
    let structural = ObservationAssumption::Structural("noninformative censoring".into());
    assert_eq!(
        digest(observed(vec![independent.clone(), structural.clone()])),
        digest(observed(vec![structural.clone(), independent.clone()]))
    );
    assert_eq!(
        digest(observed(vec![ObservationAssumption::IndependentGiven(vec![v(5), v(6)].into())])),
        digest(observed(vec![ObservationAssumption::IndependentGiven(vec![v(6), v(5)].into())]))
    );

    let randomized = |probability: f64| {
        let units: Vec<Arc<str>> = (0..4).map(|i| Arc::<str>::from(format!("unit-{i}"))).collect();
        CausalQuery::RandomizedEffect(RandomizedEffectQuery::bernoulli_itt(
            v(0),
            vec![true, false, true, false],
            vec![probability; 4],
            units.clone(),
            units,
            ("control", "treated"),
        ))
    };
    let anomaly = |max_units: usize| {
        CausalQuery::AnomalyAttribution(AnomalyAttributionQuery::new([v(1)], max_units))
    };

    // Grids, weightings, observation declarations, design arrays and attribution
    // queries each move the digest when one of their fields moves.
    let distinct = [
        digest(response(curve(vec![0.0, 1.0]))),
        digest(response(curve(vec![0.0, 2.0]))),
        digest(response(curve(vec![1.0, 0.0]))),
        digest(derivative(0.0)),
        digest(derivative(0.5)),
        digest(observed(Vec::new())),
        digest(observed(vec![structural])),
        digest(observed(vec![independent])),
        digest(randomized(0.5)),
        digest(randomized(0.25)),
        digest(anomaly(10)),
        digest(anomaly(20)),
        digest(average(Vec::new())),
        digest(distribution(Vec::new())),
    ];
    let unique: BTreeSet<_> = distinct.iter().collect();
    assert_eq!(unique.len(), distinct.len(), "every field change moves the digest");
}
