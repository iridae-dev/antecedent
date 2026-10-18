# Calibration runtime and evidence reuse

The full calibration run at `c30e10304f4bd85a01ad81dd133a360ddc95c2f1`
spent 6 h 34 min in its parallel measurement phase, excluding the release build.
Its 2,137 grid jobs consumed about 38.8 summed job-hours. There were 282
precision rechecks, accounting for approximately 35% of measured simulation
time. These are execution costs, not 2,137 independent statistical tests: jobs
can emit multiple levels or share the same simulated outcomes.

The longest job was the largest practitioner linear-adjustment bootstrap grid
point: 3 h 24 min at one simulation worker. It eventually passed. A scheduling
bug discarded the historical longest-first ordering when the measured commit
had no smoke pilot, so this job started more than three hours into the run.
The runner now retains historical longest-first ordering without requiring a
pilot at HEAD, and its tail policy can allocate additional workers to unusually
long jobs within the global worker budget. Earlier smoke timings are scheduling
estimates only; they never count as calibration evidence.

## Preserve completed work

Use `scripts/measure_calibration.sh --missing-only` to fill missing current-method
evidence while unchanged historical records await valid attestation waivers.
Explicit `--group` filters and declared `--replicates` budgets support targeted
work. Selection distinguishes current method identities from older records with
the same test name. A numerical method change requires its own measurement;
stale provenance alone does not justify repeating unchanged simulations.

Missing-only and filtered collection preserve existing records and harvest
complete passing groups even if another group fails. A failed or incomplete
grid, failed recheck, malformed payload, or source mismatch excludes the whole
group. Fresh evidence may replace an existing ID only through an explicit
replacement allowlist. Original logs and consoles remain in durable source-SHA
archives. A saved passing result is useful evidence even if the final attestation
gate still reports pending waivers.

The tally sidecars also support genuine prefix extension: retain the original
source, seed domain, grid point and raw tally state, then compute only the new
replicate indices. Conditional reported counts remain distinct from attempted
counts. An extension is not an independent confirmation, and an old-domain
extension does not acquire the newer prospective family-wise decision protocol.
Archive the selected prebuild executables with their source, command and byte
hashes before a shared target can overwrite them. Retain these small executable
receipts alongside the raw logs and tally prefixes, rather than keeping every
compiler cache. A later prefix completion can then reuse the exact producer
without rebuilding a historical checkout.

If a diagnostic array was not persisted, its printed extension median describes
the new suffix only. For an upper-median width limit, independently passing that
same limit in both the original and suffix blocks proves the combined sample
also meets it; this does not reconstruct an exact combined median.

## Budget compilation, simulation and collection separately

Batch source changes before a coalesced release prebuild. The embedded coverage
snapshot is about 2.5 MB; changing it invalidates `antecedent-io-core` and
downstream compilation. That dependency explains invalidation, but does not by
itself measure how much compilation time the snapshot causes. Compile time must
be reported separately from numerical wall time. Use a necessary build's Cargo
timings rather than trigger another build solely to time it.

A managed immutable measurement checkout prevents continuing branch work from
changing an active experiment. Give different source checkouts separate build
caches, or reserve a shared cache for the frozen source until its numerical
queue finishes. Limit the *sum* of simulation workers, independent replay and
extension workers, and competing compiler or test workers. A per-process limit
alone does not prevent oversubscription.

When two same-version checkouts share a Cargo target directory, a successful
`--no-run` alone does not prove that the executable matches the second checkout.
An observed C30/e15 switch reused the C30 binary in 0.14 seconds even though the
ranking test source differed: the second checkout's older file timestamps made
Cargo call every local artifact fresh. This was caught before any e15 simulation.
The rejected binary and source receipts were retained; only source-file timestamps
were invalidated, leaving source bytes and Git identities unchanged.

The calibration prebuild now serializes cooperating builds and fingerprints Cargo
metadata plus crate sources and assets. It advances timestamps only for changed
inputs before invoking Cargo, so older copied timestamps cannot hide different
source bytes. A cache with no source receipt is invalidated once; an unchanged
receipt or Python-only orchestration edit reuses its compiled inputs. The receipt
records each content hash under the target directory. This guard does not replace
exclusive cache ownership when using Cargo directly, and it never removes compiler
artifacts, changes numerical methods, or alters measurement provenance.

Keep the number of build caches bounded. Reuse one target directory for ordinary
branch checks and retain a separate cache only when an active frozen measurement
or attestation checkout needs it. Repeated target directories, feature sets and
test profiles can retain many generations of binaries and incremental compiler
state. Share the same profile and feature set where the checks permit it.

Before deleting a cache, check which builds and test executables are using it.
Remove only inactive compiler output; keep raw calibration logs, tally sidecars,
archived measurement binaries and provenance receipts. Do not run `cargo clean`
against an active measurement target. Once an isolated checkout has finished its
measurements and replay obligations, retire its compiler cache. Compare available
disk space before and after cleanup: APFS clones and shared blocks can make the
recovered space much smaller than the sum of directory sizes. For future
one-shot debug builds, `CARGO_INCREMENTAL=0` avoids accumulating incremental
state at the cost of slower subsequent compilation; set it before building,
rather than changing the configuration of a running build.

Collection can also become the bottleneck: repeated dependency-graph scans for
individual completed groups delayed reporting and harvesting in an earlier
targeted run. The frozen targeted controller therefore dispatches numerical work
independently from a single asynchronous collector, publishing batches of whole
passing groups approximately every minute or three completed groups. Raw logs
are durable immediately. This batching policy belongs to that targeted
controller; the ordinary wrapper still collects after its numerical run.

The runtime changes preserve nominal coverage targets, reporting floors,
usefulness limits and minimum scored replicate requirements. The separately
versioned suite-wide coverage decision protocol controls Monte Carlo false
rejections over a source-fixed family and registered looks. It neither turns a
failed legacy gate into a prospective pass nor excuses a scientific floor miss.
A fresh seed domain and frozen source are required for new prospective evidence;
old accepted evidence retains its original source and decision semantics.
