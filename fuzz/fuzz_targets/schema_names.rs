//! Fuzz CausalSchemaBuilder name uniqueness and dense ID assignment.
#![no_main]

use causal_core::{
    CausalSchemaBuilder, MeasurementSpec, SmallRoleSet, ValueType,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut builder = CausalSchemaBuilder::new();
    for chunk in data.chunks(8) {
        if chunk.is_empty() {
            continue;
        }
        let name = String::from_utf8_lossy(chunk).into_owned();
        if name.is_empty() {
            continue;
        }
        let _ = builder.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::empty(),
            None,
            None,
            MeasurementSpec::default(),
        );
    }
    let _ = builder.build();
});
