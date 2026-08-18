//! Workspace-level invariants that are cheap to gate and easy to regress.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;
use std::path::{Path, PathBuf};

/// Direct `rayon` in a workspace `Cargo.toml` would introduce a global thread
/// pool outside [`antecedent_core::ExecutionContext`]. Transitive `rayon` in
/// `Cargo.lock` (via faer / criterion / …) is not this gate.
#[test]
fn no_direct_rayon_dependency() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/antecedent → workspace root");
    let mut offenders = Vec::new();
    for cargo in workspace_cargo_tomls(workspace) {
        let text =
            fs::read_to_string(&cargo).unwrap_or_else(|e| panic!("read {}: {e}", cargo.display()));
        if declares_direct_rayon(&text) {
            offenders.push(cargo);
        }
    }
    assert!(
        offenders.is_empty(),
        "direct `rayon` dependency is rejected (ExecutionContext owns parallelism): {}",
        offenders
            .iter()
            .map(|p| p.strip_prefix(workspace).unwrap_or(p).display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
}

fn workspace_cargo_tomls(root: &Path) -> Vec<PathBuf> {
    let mut out = vec![root.join("Cargo.toml")];
    if let Ok(entries) = fs::read_dir(root.join("crates")) {
        for ent in entries.flatten() {
            let cargo = ent.path().join("Cargo.toml");
            if cargo.is_file() {
                out.push(cargo);
            }
        }
    }
    let python = root.join("python/Cargo.toml");
    if python.is_file() {
        out.push(python);
    }
    out
}

fn declares_direct_rayon(toml: &str) -> bool {
    for line in toml.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("rayon") {
            let rest = rest.trim_start();
            if rest.starts_with('=') || rest.starts_with('.') {
                return true;
            }
        }
    }
    false
}
