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

const FAMILIES: [&str; 5] =
    ["static_effect", "static_response", "structural", "temporal", "design_based"];

#[test]
fn every_route_cell_is_registered_once_to_a_family() {
    let mut names = BTreeSet::new();
    for (cell, family) in REGISTERED {
        assert!(FAMILIES.contains(family), "{} is registered to {family}", cell.as_str());
        assert!(names.insert(cell.as_str()), "{} is registered twice", cell.as_str());
    }
    assert_eq!(names.len(), 45, "route cell inventory drifted");
    for cell in GraphPosteriorCell::ALL {
        assert!(names.insert(RouteCell::GraphPosterior(cell).as_str()), "{}", cell.as_str());
    }
}

fn linear_study() -> Study {
    let schema = CausalSchemaBuilder::new()
        .continuous("t")
        .treatment()
        .continuous("y")
        .outcome()
        .build()
        .unwrap();
    let treatment = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let outcome = [1.0, 3.1, 0.9, 2.9, 1.0, 3.0];
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
    assert_eq!(lowered.depth(), LoweringDepth::Generic);
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
