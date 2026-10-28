//! Link assumptions and refused declarations of the multi-dataset constraints.
//!
//! `oracle_data/multi_dataset_constraints_reference.py` (frozen in the JSON beside it) writes out
//! the allowed and forbidden directed links for every pair of the five node roles at lags 0 to 3.
//! The product's `gunther_forbids` must agree on all 100 rows, a dummy must take precedence over
//! a context of the same id, an unlisted id must be a system variable, and the declarations the
//! pooled search does not read (a cross-environment link assumption, the lagged-pooling switch)
//! must be refused by the search rather than accepted and ignored.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    VariableId,
};
use antecedent_data::{
    Float64Column, MultiEnvironmentData, OwnedColumn, OwnedColumnarStorage, SamplingRegularity,
    TimeIndex, TimeSeriesData, ValidityBitmap,
};
use antecedent_discovery::LaggedLink;
use antecedent_discovery::{
    ContextKind, CrossEnvLinkAssumption, DiscoveryConstraints, DiscoveryWorkspace, JpcmciNodeRole,
    JpcmciPlus, MultiDatasetConstraints, TemporalConstraints,
};
use serde_json::Value;

fn id(role: &str) -> VariableId {
    VariableId::from_raw(match role {
        "system" => 0,
        "space_context" => 1,
        "time_context" => 2,
        "space_dummy" => 3,
        "time_dummy" => 4,
        other => panic!("unknown role {other}"),
    })
}

fn declared() -> MultiDatasetConstraints {
    MultiDatasetConstraints {
        context_variables: Arc::from([id("space_context"), id("time_context")]),
        context_kinds: Arc::from([(id("time_context"), ContextKind::Time)]),
        space_dummy_variables: Arc::from([id("space_dummy")]),
        time_dummy_variables: Arc::from([id("time_dummy")]),
        ..MultiDatasetConstraints::default()
    }
}

#[test]
fn the_link_assumption_table_equals_the_written_out_rules() {
    let rows: Vec<Value> =
        serde_json::from_str(include_str!("oracle_data/multi_dataset_constraints_reference.json"))
            .unwrap();
    assert_eq!(rows.len(), 100);
    let md = declared();
    for row in &rows {
        let (src, tgt) = (row["source"].as_str().unwrap(), row["target"].as_str().unwrap());
        let lag = row["lag"].as_u64().unwrap() as u32;
        let link = LaggedLink {
            source: id(src),
            source_lag: Lag::from_raw(lag),
            target: id(tgt),
            target_lag: Lag::CONTEMPORANEOUS,
        };
        assert_eq!(
            md.gunther_forbids(link),
            row["forbidden"].as_bool().unwrap(),
            "{src}({lag}) -> {tgt}"
        );
    }
    assert_eq!(md.role_of(id("system")), JpcmciNodeRole::System);
    assert_eq!(md.role_of(id("space_context")), JpcmciNodeRole::SpaceContext);
    assert_eq!(md.role_of(id("time_context")), JpcmciNodeRole::TimeContext);
    assert_eq!(md.role_of(id("space_dummy")), JpcmciNodeRole::SpaceDummy);
    assert_eq!(md.role_of(id("time_dummy")), JpcmciNodeRole::TimeDummy);
    assert_eq!(md.role_of(VariableId::from_raw(17)), JpcmciNodeRole::System);
    // A synthetic dummy that carries the id of a declared context is the dummy.
    let both = MultiDatasetConstraints {
        context_variables: Arc::from([VariableId::from_raw(5)]),
        space_dummy_variables: Arc::from([VariableId::from_raw(5)]),
        ..MultiDatasetConstraints::default()
    };
    assert_eq!(both.role_of(VariableId::from_raw(5)), JpcmciNodeRole::SpaceDummy);
}

fn two_environments() -> MultiEnvironmentData {
    let environment = |offset: usize| {
        let mut schema = CausalSchemaBuilder::new();
        for i in 0..2 {
            schema
                .add_variable(
                    format!("v{i}").as_str(),
                    ValueType::Continuous,
                    SmallRoleSet::from_hint(RoleHint::Context),
                    None,
                    None,
                    MeasurementSpec::default(),
                )
                .unwrap();
        }
        let columns = (0..2)
            .map(|i| {
                let values: Vec<f64> = (0..40)
                    .map(|t| (((t + offset) * (i + 3)) % 11) as f64 + 0.01 * t as f64)
                    .collect();
                OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw(i as u32),
                        Arc::from(values),
                        ValidityBitmap::all_valid(40),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let storage =
            OwnedColumnarStorage::try_new(schema.build().unwrap(), columns, None, None).unwrap();
        TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: 40 },
        )
        .unwrap()
    };
    MultiEnvironmentData::try_new(Arc::from([environment(0), environment(3)])).unwrap()
}

#[test]
fn declarations_the_pooled_search_does_not_read_are_refused_by_the_search() {
    let multi = two_environments();
    let variables = [VariableId::from_raw(0), VariableId::from_raw(1)];
    let run = |multi_dataset: MultiDatasetConstraints| {
        JpcmciPlus::new()
            .with_fdr(false)
            .with_constraints(DiscoveryConstraints {
                temporal: TemporalConstraints {
                    min_lag: Lag::CONTEMPORANEOUS,
                    max_lag: Lag::from_raw(1),
                },
                multi_dataset,
                ..DiscoveryConstraints::default()
            })
            .run(
                &multi,
                &variables,
                &mut DiscoveryWorkspace::default(),
                &ExecutionContext::for_tests(3),
            )
    };
    run(MultiDatasetConstraints::default()).expect("the default declarations run");
    for cross_env in [
        CrossEnvLinkAssumption::SharedContemporaneousSkeleton,
        CrossEnvLinkAssumption::EnvironmentSpecificContemporaneous,
    ] {
        let error =
            run(MultiDatasetConstraints { cross_env, ..MultiDatasetConstraints::default() })
                .expect_err("a cross-environment link assumption is not read by the search");
        assert!(error.to_string().contains("cross-environment"), "{error}");
    }
    let error = run(MultiDatasetConstraints {
        pool_lagged_ci: false,
        ..MultiDatasetConstraints::default()
    })
    .expect_err("pooling cannot be switched off");
    assert!(error.to_string().contains("pool_lagged_ci"), "{error}");
}
