//! Fuzz causal query wire: CBOR bytes, then the domain from-wire decode.
#![no_main]

use antecedent_io::{CausalQueryWire, causal_query_from_wire, from_cbor};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(wire) = from_cbor::<CausalQueryWire>(data) {
        let _ = causal_query_from_wire(&wire);
    }
});
