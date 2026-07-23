//! Fuzz Python-boundary Rust helpers (DOT parse + schema names).
#![no_main]

use antecedent_core::{CausalSchemaBuilder, MeasurementSpec, SmallRoleSet, ValueType};
use antecedent_io::dag_from_dot;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        // Cap length so pathological DOT inputs stay bounded.
        let slice = if s.len() > 4096 { &s[..4096] } else { s };
        let _ = dag_from_dot(slice);
    }
    let mut b = CausalSchemaBuilder::new();
    for chunk in data.chunks(8).take(32) {
        if chunk.is_empty() {
            continue;
        }
        let name = String::from_utf8_lossy(chunk);
        if name.is_empty() {
            continue;
        }
        let _ = b.add_variable(
            name.as_ref(),
            ValueType::Continuous,
            SmallRoleSet::empty(),
            None,
            None,
            MeasurementSpec::default(),
        );
    }
    let _ = b.build();
});
