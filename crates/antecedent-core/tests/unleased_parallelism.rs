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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use antecedent_core::ExecutionContext;
use antecedent_core::execution::{AdmissionError, AdmissionPool, PipelineStep};

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
        path: "crates/antecedent-core/src/execution/pipeline.rs",
        marker: "acquire_map_lease",
        reason: "the admission pool's own parallel map: it takes the worker lease before it \
                 spawns and holds it until the workers join",
    },
    Allowed {
        path: "crates/antecedent-data/src/resample/replicates.rs",
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
        } else if path.extension().is_some_and(|ext| ext == "rs")
            && !name.ends_with("tests.rs")
            && name != "tests.rs"
        {
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

/// Runs `work` on `threads` plain test threads at once.
fn concurrently(threads: usize, work: impl Fn(usize) + Sync) {
    std::thread::scope(|scope| {
        for id in 0..threads {
            let work = &work;
            scope.spawn(move || work(id));
        }
    });
}

#[test]
fn concurrent_allocation_never_exceeds_the_pool_bytes_and_refusals_leave_it_whole() {
    const POOL_BYTES: u64 = 1 << 20;
    const HOLD: usize = 256 << 10;
    let pool = AdmissionPool::fixed(4, POOL_BYTES);
    let ctx = ExecutionContext::production(1, 4).with_admission(pool.clone());

    // Four holds fill the pool exactly; the next is refused with the remainder it saw.
    let held: Vec<_> = (0..4).map(|_| ctx.acquire_admission(0, HOLD as u64).unwrap()).collect();
    let refused = ctx.acquire_admission(0, 1).unwrap_err();
    assert!(
        matches!(refused, AdmissionError::ExceedsRemainder { remaining_bytes: 0, .. }),
        "{refused}"
    );
    drop(held);
    assert_eq!(pool.available().bytes(), POOL_BYTES);

    // Many threads race for the same bytes. Every grant allocates for real, and the sum of
    // live allocations is observed from inside the holders.
    let live = AtomicUsize::new(0);
    let peak = AtomicUsize::new(0);
    let granted = AtomicUsize::new(0);
    let refusals = AtomicUsize::new(0);
    concurrently(16, |_| {
        for round in 0..40 {
            match ctx.acquire_admission(0, HOLD as u64) {
                Ok(lease) => {
                    let buffer = vec![round as u8; HOLD];
                    let now = live.fetch_add(buffer.len(), Ordering::SeqCst) + buffer.len();
                    peak.fetch_max(now, Ordering::SeqCst);
                    granted.fetch_add(1, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_micros(300));
                    assert_eq!(buffer[HOLD - 1], round as u8);
                    live.fetch_sub(buffer.len(), Ordering::SeqCst);
                    drop(buffer);
                    drop(lease);
                }
                Err(AdmissionError::ExceedsRemainder { .. }) => {
                    refusals.fetch_add(1, Ordering::SeqCst);
                }
                Err(other) => panic!("unexpected admission error: {other}"),
            }
        }
    });
    let observed = peak.load(Ordering::SeqCst);
    assert!(observed as u64 <= POOL_BYTES, "live allocation {observed} passed the pool");
    assert!(granted.load(Ordering::SeqCst) > 0);
    assert!(refusals.load(Ordering::SeqCst) > 0, "contention must produce refusals");
    assert_eq!(live.load(Ordering::SeqCst), 0);
    assert_eq!(pool.available().bytes(), POOL_BYTES, "refusals and drops leave the pool whole");
    assert_eq!(pool.available().threads(), 4);
}

#[test]
fn parallel_maps_run_no_more_workers_than_the_pool_and_scratch_stays_inside_its_bytes() {
    let pool = AdmissionPool::fixed(3, 1 << 20);
    let ctx = ExecutionContext::production(1, 8).with_admission(pool.clone());
    let running = AtomicUsize::new(0);
    let peak = AtomicUsize::new(0);
    let scratch_live = AtomicUsize::new(0);
    let scratch_peak = AtomicUsize::new(0);
    let out = ctx
        .map_indexed_with_scratch(
            64,
            || {
                let now = scratch_live.fetch_add(64 << 10, Ordering::SeqCst) + (64 << 10);
                scratch_peak.fetch_max(now, Ordering::SeqCst);
                vec![0_u8; 64 << 10]
            },
            |index, _, scratch| {
                let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                scratch[0] = index as u8;
                std::thread::sleep(Duration::from_micros(500));
                running.fetch_sub(1, Ordering::SeqCst);
                Ok::<_, ()>(index * 2)
            },
        )
        .unwrap();
    assert_eq!(out, (0..64).map(|i| i * 2).collect::<Vec<_>>());
    assert!(peak.load(Ordering::SeqCst) <= 3, "workers ran past the pool: {peak:?}");
    assert!(
        scratch_peak.load(Ordering::SeqCst) <= 3 * (64 << 10),
        "one scratch per leased worker: {scratch_peak:?}"
    );
    assert_eq!(pool.available().threads(), 3);

    // Two studies share one pool: together they never run more workers than it holds.
    let together = AtomicUsize::new(0);
    let together_peak = AtomicUsize::new(0);
    concurrently(4, |_| {
        let study = ExecutionContext::production(2, 8).with_admission(pool.clone());
        let _ = study.map_indexed(32, |_, _| {
            let now = together.fetch_add(1, Ordering::SeqCst) + 1;
            together_peak.fetch_max(now, Ordering::SeqCst);
            std::thread::sleep(Duration::from_micros(300));
            together.fetch_sub(1, Ordering::SeqCst);
            Ok::<_, ()>(())
        });
    });
    // Callers that find no free slot run on their own thread, so the pool's three workers
    // plus at most one inline caller per study bound the concurrency.
    assert!(together_peak.load(Ordering::SeqCst) <= 3 + 4, "{together_peak:?}");
    assert_eq!(pool.available().threads(), 3);
}

#[test]
fn a_pipeline_keeps_live_buffers_at_the_window_for_a_slow_consumer() {
    let pool = AdmissionPool::fixed(4, 1 << 20);
    let ctx = ExecutionContext::production(1, 4).with_admission(pool.clone());
    let live = AtomicUsize::new(0);
    let peak = AtomicUsize::new(0);
    let mut total = 0_u64;
    let receipt = ctx
        .pipeline_indexed(
            200,
            5,
            |index, _| {
                let now = live.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                Ok::<_, ()>(vec![index as u8; 4096])
            },
            |_, buffer| {
                std::thread::sleep(Duration::from_micros(200));
                total += buffer.iter().map(|&b| u64::from(b)).sum::<u64>();
                live.fetch_sub(1, Ordering::SeqCst);
                Ok(PipelineStep::Continue)
            },
        )
        .unwrap();
    assert!(peak.load(Ordering::SeqCst) <= 5, "{peak:?}");
    assert!(receipt.peak_live <= 5);
    assert_eq!(receipt.items_consumed, 200);
    assert_eq!(total, (0..200_u64).map(|i| (i % 256) * 4096).sum::<u64>());
    assert_eq!(pool.available().threads(), 4);
}
