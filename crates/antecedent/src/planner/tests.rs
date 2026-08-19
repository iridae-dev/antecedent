use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, CausalQuery, DataClassification, ExecutionContext,
    LogicalAnalysisPlanRecord, MemoryBudget, TemporalEffectQuery, VariableId,
};
use antecedent_data::{DiscoveryEstimationSplit, TabularData, TimeSeriesData};
use antecedent_graph::{Dag, TemporalDag};

use crate::accepted::AcceptedGraph;
use crate::error::CausalError;
use crate::strategy_table::IdentifierId;

use super::*;

    fn tabular_plan(rows: u64) -> LogicalAnalysisPlan {
        let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        LogicalAnalysisPlan {
            record: LogicalAnalysisPlanRecord {
                plan_id: Arc::from("test"),
                data_classification: DataClassification::Tabular,
                discovery_algorithm: None,
                graph_review_required: false,
                identifier: Some(Arc::from("backdoor.adjustment")),
                estimator: Some(Arc::from("linear.adjustment.ate")),
                validation_suite: None,
                query_variables: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
            },
            query: CausalQuery::AverageEffect(q),
            split: None,
            row_count_hint: rows,
        }
    }

    #[test]
    fn static_ate_compiles_with_schedule_and_copies() {
        let plan = tabular_plan(200);
        plan.validate().unwrap();
        let ctx = ExecutionContext::for_tests(1);
        let physical = plan.compile_physical(&ctx).unwrap();
        assert!(physical.record.estimated_peak_memory_bytes.is_some());
        assert_eq!(physical.record.kernels.len(), 1);
        assert_eq!(physical.record.estimated_copy_bytes, Some(200 * 8 * 8));
        assert_eq!(physical.record.task_schedule.len(), 1);
        assert_eq!(&*physical.record.task_schedule[0].dimension, "serial");
        assert!(!physical.record.materializations.is_empty());
    }

    #[test]
    fn temporal_query_on_tabular_fails() {
        let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0);
        let plan = LogicalAnalysisPlan {
            record: LogicalAnalysisPlanRecord {
                plan_id: Arc::from("bad"),
                data_classification: DataClassification::Tabular,
                discovery_algorithm: None,
                graph_review_required: false,
                identifier: None,
                estimator: None,
                validation_suite: None,
                query_variables: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
            },
            query: CausalQuery::TemporalEffect(q),
            split: None,
            row_count_hint: 10,
        };
        assert!(matches!(plan.validate(), Err(CausalError::Compile { .. })));
    }

    #[test]
    fn pcmci_on_tabular_fails() {
        let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let plan = LogicalAnalysisPlan {
            record: LogicalAnalysisPlanRecord {
                plan_id: Arc::from("bad"),
                data_classification: DataClassification::Tabular,
                discovery_algorithm: Some(Arc::from("pcmci")),
                graph_review_required: true,
                identifier: None,
                estimator: None,
                validation_suite: None,
                query_variables: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
            },
            query: CausalQuery::AverageEffect(q),
            split: None,
            row_count_hint: 10,
        };
        assert!(matches!(plan.validate(), Err(CausalError::Compile { .. })));
    }

    #[test]
    fn soft_memory_limit_refuses_dense_plan() {
        let plan = tabular_plan(10_000);
        let mut ctx = ExecutionContext::for_tests(1);
        ctx.memory = MemoryBudget { soft_limit_bytes: Some(64), hard_limit_bytes: None };
        assert!(matches!(plan.compile_physical(&ctx), Err(CausalError::Resource { .. })));
    }

    #[test]
    fn split_row_hint_drives_batch_size() {
        let mut plan = tabular_plan(100);
        plan.split = Some(DiscoveryEstimationSplit::from_sizes(100, 50, 10, 40).unwrap());
        plan.row_count_hint = 40;
        let ctx = ExecutionContext::for_tests(1);
        let physical = plan.compile_physical(&ctx).unwrap();
        assert_eq!(physical.record.batch_size, Some(40));
    }

    fn toy_static_input() -> (TabularData, Dag, AverageEffectQuery) {
        use antecedent_core::{
            CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
        };
        use antecedent_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, ValidityBitmap};
        use antecedent_graph::DenseNodeId;
        use std::sync::Arc as StdArc;

        let n = 10usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "t",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let t: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
        let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * t[i]).collect();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    StdArc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    StdArc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        (TabularData::new(storage), dag, query)
    }

    #[test]
    fn refuses_iv_estimator_with_backdoor_identifier() {
        let (data, graph, query) = toy_static_input();
        let err = compile_logical_static_ate(StaticAteCompileInput {
            data: &data,
            graph: &graph,
            query: &query,
            validation_suite: None,
            identifier: Arc::from("backdoor.adjustment"),
            estimator: Arc::from("iv.2sls"),
        })
        .unwrap_err();
        assert!(matches!(err, CausalError::Compile { .. }));
    }

    #[test]
    fn refuses_propensity_estimator_with_frontdoor_identifier() {
        let (data, graph, query) = toy_static_input();
        let err = compile_logical_static_ate(StaticAteCompileInput {
            data: &data,
            graph: &graph,
            query: &query,
            validation_suite: None,
            identifier: Arc::from("frontdoor"),
            estimator: Arc::from("propensity.weighting"),
        })
        .unwrap_err();
        assert!(matches!(err, CausalError::Compile { .. }));
    }

    #[test]
    fn refuses_unknown_identifier_and_estimator() {
        let (data, graph, query) = toy_static_input();
        let err = compile_logical_static_ate(StaticAteCompileInput {
            data: &data,
            graph: &graph,
            query: &query,
            validation_suite: None,
            identifier: Arc::from("backdoor.adjustment"),
            estimator: Arc::from("not.a.real.estimator"),
        })
        .unwrap_err();
        assert!(matches!(err, CausalError::Compile { .. }));
    }

    #[test]
    fn refuses_att_target_population_with_linear_adjustment() {
        use antecedent_core::TargetPopulation;
        let (data, graph, query) = toy_static_input();
        let att_query = query.with_target_population(TargetPopulation::Treated);
        let err = compile_logical_static_ate(StaticAteCompileInput {
            data: &data,
            graph: &graph,
            query: &att_query,
            validation_suite: None,
            identifier: Arc::from("backdoor.adjustment"),
            estimator: Arc::from("linear.adjustment.ate"),
        })
        .unwrap_err();
        assert!(matches!(err, CausalError::Compile { .. }));
    }

    #[test]
    fn refuses_planned_target_population_on_temporal_effect() {
        use antecedent_core::{
            CausalSchemaBuilder, MeasurementSpec, PredicateExpr, RoleHint, SmallRoleSet,
            TargetPopulation, ValueType,
        };
        use antecedent_data::{
            Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
            ValidityBitmap,
        };
        use std::sync::Arc as StdArc;

        let n = 8usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    StdArc::from(vec![0.0; n]),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    StdArc::from(vec![0.0; n]),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap();
        let graph = TemporalDag::empty();
        let query =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_target_population(TargetPopulation::Predicate(PredicateExpr::named(
                    "cohort_a",
                )));
        let err = compile_logical_temporal_effect(&data, &graph, &query, None, false).unwrap_err();
        assert!(matches!(err, CausalError::Compile { .. }));
    }

    #[test]
    fn refuses_temporal_query_vars_not_in_temporal_dag() {
        use antecedent_core::{
            CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
        };
        use antecedent_data::{
            Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
            ValidityBitmap,
        };
        use std::sync::Arc as StdArc;

        let n = 8usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    StdArc::from(vec![0.0; n]),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    StdArc::from(vec![0.0; n]),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap();
        // A populated temporal DAG that does NOT name the query variables: the plan must
        // fail at compile time rather than silently accepting an unfounded query
        // (target_population is left at the default AllObserved so the later population
        // check cannot be the one firing).
        //
        // The DAG must be non-empty: a node-less TemporalDag is the placeholder the
        // graph-posterior path supplies, and is deliberately exempt from this check —
        // see `validate_query_vars_in_temporal_dag`.
        let mut graph = TemporalDag::empty();
        graph.add_lagged(VariableId::from_raw(7), antecedent_core::Lag::CONTEMPORANEOUS).unwrap();
        graph.add_lagged(VariableId::from_raw(8), antecedent_core::Lag::CONTEMPORANEOUS).unwrap();
        let query =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0);
        let err = compile_logical_temporal_effect(&data, &graph, &query, None, false).unwrap_err();
        let CausalError::Compile { message } = err else {
            panic!("expected CausalError::Compile, got {err:?}");
        };
        assert!(
            message.contains("not in temporal DAG"),
            "expected a temporal-DAG membership error, got: {message}"
        );
    }

    #[test]
    fn accepts_default_pair() {
        let (data, graph, query) = toy_static_input();
        let plan = compile_logical_static_ate(StaticAteCompileInput {
            data: &data,
            graph: &graph,
            query: &query,
            validation_suite: None,
            identifier: Arc::from("backdoor.adjustment"),
            estimator: Arc::from("linear.adjustment.ate"),
        })
        .unwrap();
        assert_eq!(plan.record.identifier.as_deref(), Some("backdoor.adjustment"));
        assert_eq!(plan.record.estimator.as_deref(), Some("linear.adjustment.ate"));
    }

    #[test]
    fn accepts_propensity_weighting_with_backdoor_adjustment() {
        let (data, graph, query) = toy_static_input();
        let plan = compile_logical_static_ate(StaticAteCompileInput {
            data: &data,
            graph: &graph,
            query: &query,
            validation_suite: None,
            identifier: Arc::from("backdoor.adjustment"),
            estimator: Arc::from("propensity.weighting"),
        })
        .unwrap();
        assert_eq!(plan.record.estimator.as_deref(), Some("propensity.weighting"));
    }

    #[test]
    fn refuses_dag_only_identifier_on_pag() {
        use antecedent_graph::Pag;
        let structure = AcceptedGraph::pag(Pag::with_variables(2));
        let err = reject_dag_only_on_pag(&structure, IdentifierId::BackdoorAdjustment).unwrap_err();
        assert!(matches!(err, CausalError::Compile { .. }));
        // Class-aware identifier is allowed through this gate.
        let structure = AcceptedGraph::pag(Pag::with_variables(2));
        reject_dag_only_on_pag(&structure, IdentifierId::GeneralizedAdjustment).unwrap();
    }
