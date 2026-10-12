//! Fuzz the byte-slice artifact reader.
//!
//! `mmap_file::map_file_readonly` only maps a filesystem `File`. This target
//! uses `MappedArtifactReader::from_bytes`, which indexes the same container
//! framing from an owned snapshot and does not touch the filesystem.
#![no_main]

use std::sync::Arc;

use antecedent_io::MappedArtifactReader;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = MappedArtifactReader::from_bytes(Arc::<[u8]>::from(data));
});
