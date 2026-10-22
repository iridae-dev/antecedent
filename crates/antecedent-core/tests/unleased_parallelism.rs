// SPDX-License-Identifier: MIT OR Apache-2.0
//! Static guard: production code does not start threads outside the admission pool.
//!
//! Every parallel region must lease its workers and scratch from the study's admission pool
//! (`ExecutionContext::map_indexed`, `admit_scratch`, `acquire_admission`). This test scans the
//! production sources of every workspace crate and of the Python extension for direct thread
//! creation and parallel-iterator libraries, and fails for any file that is not on the
//! allowlist below. Test code (`tests/` directories, `*tests.rs` files, and inline
//! `#[cfg(test)] mod ... {` blocks) and comments are not scanned.
//!
//! A new site that needs a thread goes through `map_indexed`. A file is added to the allowlist
//! only with a reason that says why the thread is already accounted for, and with a marker the
//! file must keep so the reason cannot silently go stale.
use std::fs;
use std::path::{Path, PathBuf};

/// Text that starts a thread or hands work to a parallel-iterator library.
const FORBIDDEN: &[&str] = &[
    "thread::spawn",
    "thread::scope",
    "thread::Builder",
    "spawn_scoped",
    "rayon::",
    "Par::rayon",
    "par_iter",
    "into_par_iter",
    "par_chunks",
];

/// A production file that may start threads, why, and text it must keep.
struct Allowed {
    path: &'static str,
    /// The file must still contain this text, which is what makes the reason true.
    marker: &'static str,
    reason: &'static str,
}

const ALLOWLIST: &[Allowed] = &[
    Allowed {
        path: "crates/antecedent-core/src/execution.rs",
        marker: "acquire_map_lease",
        reason: "the admission pool's own parallel map: it takes the worker lease before it \
                 spawns and holds it until the workers join",
    },
    Allowed {
        path: "crates/antecedent-data/src/resample.rs",
        marker: "admit_resample_scratch",
        reason: "batch resampling leases its workers and per-worker scratch with \
                 `admit_scratch` before the scope starts, and the scope uses exactly the \
                 granted worker count",
    },
    Allowed {
        path: "crates/antecedent/src/analysis/flagship_mixed.rs",
        marker: "ROUTE_STACK_BYTES",
        reason: "one scoped thread that only gives the route a larger stack; the caller \
                 blocks on it, so it adds no parallelism, and the route's work admits itself",
    },
    Allowed {
        path: "python/src/interrupt.rs",
        marker: "WORKER_STACK_BYTES",
        reason: "one worker so the calling thread can poll Python signals and release the \
                 GIL; the same work runs serially there and admits itself",
    },
];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("antecedent-core sits two levels below the workspace root")
        .to_path_buf()
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if path.is_dir() {
            if matches!(name.as_str(), "tests" | "benches" | "examples" | "target") {
                continue;
            }
            rust_files(&path, out);
        } else if name.ends_with(".rs") && !name.ends_with("tests.rs") && name != "tests.rs" {
            out.push(path);
        }
    }
}

/// The production lines of `source`: comments are dropped, and scanning stops at the first
/// inline test module (`#[cfg(test)]` immediately followed by a `mod name {` block).
fn production_lines(source: &str) -> Vec<(usize, &str)> {
    let lines: Vec<&str> = source.lines().collect();
    let mut out = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("#[cfg(test)]") {
            let opens_block = lines.get(index + 1).is_some_and(|next| {
                let next = next.trim_start();
                (next.starts_with("mod ") || next.starts_with("pub(crate) mod "))
                    && next.trim_end().ends_with('{')
            });
            if opens_block {
                break;
            }
        }
        if trimmed.starts_with("//") {
            continue;
        }
        out.push((index + 1, *line));
    }
    out
}

#[test]
fn production_code_starts_no_thread_outside_the_admission_pool() {
    let root = workspace_root();
    let mut files = Vec::new();
    let crates = root.join("crates");
    for entry in fs::read_dir(&crates).expect("crates directory").flatten() {
        rust_files(&entry.path().join("src"), &mut files);
    }
    rust_files(&root.join("python").join("src"), &mut files);
    assert!(files.len() > 100, "the scan found only {} source files", files.len());

    let mut violations = Vec::new();
    let mut seen_allowed = vec![false; ALLOWLIST.len()];
    for file in &files {
        let relative = file
            .strip_prefix(&root)
            .expect("scanned files sit under the workspace root")
            .to_string_lossy()
            .replace('\\', "/");
        let source = fs::read_to_string(file).expect("source file is readable");
        let hits: Vec<(usize, &str)> = production_lines(&source)
            .into_iter()
            .filter(|(_, line)| FORBIDDEN.iter().any(|pattern| line.contains(pattern)))
            .collect();
        if hits.is_empty() {
            continue;
        }
        match ALLOWLIST.iter().position(|allowed| allowed.path == relative) {
            Some(position) => {
                seen_allowed[position] = true;
                let allowed = &ALLOWLIST[position];
                assert!(
                    source.contains(allowed.marker),
                    "{relative} starts threads but no longer contains `{}`, which is why it is \
                     allowed: {}",
                    allowed.marker,
                    allowed.reason
                );
            }
            None => {
                for (line, text) in hits {
                    violations.push(format!("{relative}:{line}: {}", text.trim()));
                }
            }
        }
    }
    assert!(
        violations.is_empty(),
        "production code starts threads without an admission lease. Route the work through \
         `ExecutionContext::map_indexed` (or lease with `admit_scratch`/`acquire_admission` \
         and justify the file in this test's allowlist):\n{}",
        violations.join("\n")
    );
    for (allowed, seen) in ALLOWLIST.iter().zip(seen_allowed) {
        assert!(
            seen,
            "{} is on the allowlist but starts no thread; remove the entry ({})",
            allowed.path, allowed.reason
        );
    }
}
