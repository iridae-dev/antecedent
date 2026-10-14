//! Fuzz the design ranking artifact wire decode.
#![no_main]

use antecedent_design::design_ranking_artifact::DesignRankingArtifactWire;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = DesignRankingArtifactWire::from_bytes(data);
});
