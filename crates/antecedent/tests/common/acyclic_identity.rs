//! Shared static linear ATE declaration for the acyclic identity witnesses.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{AverageEffectQuery, CausalSchemaBuilder};
use antecedent_data::TabularData;

use super::{Dag, RefuteSuite, Study};

pub(super) const RELEASED_STATISTICAL_KEY: &str = "static_linear_ate.lowered_statistical_v2";

pub(super) fn released_statistical_golden() -> &'static str {
    include_str!("../fixtures/acyclic_identity_goldens.txt")
        .lines()
        .filter_map(|line| line.split_once('='))
        .find_map(|(name, digest)| (name == RELEASED_STATISTICAL_KEY).then_some(digest))
        .expect("the released statistical digest is pinned in the acyclic identity golden")
}

pub(super) fn linear_ate() -> (Study, TabularData) {
    let schema = CausalSchemaBuilder::new()
        .continuous("t")
        .treatment()
        .continuous("y")
        .outcome()
        .build()
        .unwrap();
    let treatment = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let outcome = [1.0, 3.1, 0.9, 2.9, 1.0, 3.0, 1.1, 3.2];
    let columns: [(&str, &[f64]); 2] = [("t", &treatment), ("y", &outcome)];
    let data = TabularData::try_from_schema_f64(schema.clone(), columns).unwrap();
    let graph = Dag::from_named_edges(&schema, &[("t", "y")]).unwrap();
    let query =
        AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap());
    let study = Study::tabular(data.clone())
        .graph(graph)
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    (study, data)
}
