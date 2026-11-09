# Calibration runtime and evidence reuse

For calibration's role in the wider evidence system, read [How Antecedent checks its scientific methods](guides/scientific-verification.md#measure-uncertainty-as-a-complete-procedure). This page explains how measurements and their supporting artifacts are retained and reused.

A full calibration run spends hours in its parallel measurement phase, and
precision rechecks can take a large share of the simulation time. Grid jobs are
units of execution cost, not independent statistical tests: a job can emit
several levels or share the same simulated outcomes.

A few long grid jobs, such as the largest practitioner linear-adjustment
bootstrap grid points, set the wall time of a run. The runner therefore starts
the longest jobs first, ordered by their most recent recorded wall times and
falling back to smoke-pilot timings, so no pilot at HEAD is required, and its
tail policy can allocate additional workers to unusually long jobs within the
global worker budget. Pilot and earlier smoke timings are scheduling estimates
only; they never count as calibration evidence.

## Preserve completed work

Use `scripts/measure_calibration.sh --missing-only` to fill missing current-method
evidence while unchanged historical records await a probe or a re-measurement.
Explicit `--group` filters and declared `--replicates` budgets support targeted
work. Selection distinguishes current method identities from older records with
the same test name. A numerical method change requires its own measurement;
stale provenance alone does not justify repeating unchanged simulations.

Missing-only and filtered collection preserve existing records and harvest
complete passing groups even if another group fails. A failed or incomplete
grid, failed recheck, malformed payload, or source mismatch excludes the whole
group. Fresh evidence may replace an existing ID only through an explicit
replacement allowlist. Original logs and consoles remain in durable source-SHA
archives. A saved passing result is useful evidence even if the calibration
status still reports records that owe a probe or a measurement.

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

## Keep records current with probes

A record stands while the numbers it measured are unchanged, and a short re-run
can show that. Each measurement stores, per grid point, the chained digests of its
first 10 / 32 / 100 / 320 / 1000 replicates. The harness folds every replicate's
interval endpoints, truth and bound scope facts, as raw bits in a fixed order, into
a fingerprint; the `calibration-record` line and the tally sidecar carry the
prefix digests, and the collector stores them as the record's `probe` table.
Replicate `k` is a deterministic function of `k`, so
`scripts/calibration_probe.py` re-runs a record's group at
`ANTECEDENT_CALIBRATION_NSIM=k` (smoke mode, the measurement's frozen decision
environment, a clean commit) and compares the prefix digests bit for bit.

- `quick` probes `k = 10` at every grid point: seconds of simulation per group,
  a detector for a change that moved a number.
- `bounded` probes the smallest stored prefix `k >= 3/margin`, where the margin is
  the observed coverage minus the acceptance floor at the record's worst grid
  point. A match for `k` replicates bounds the per-replicate rate of an undetected
  change at `3/k`, which the margin covers. A record whose margin needs more than
  1000 replicates, or that stores no digests, is `measurement_owed` and takes a
  measurement.

A probe costs `k` of the 400 first-pass replicates (up to 2000 with the recheck)
plus the group's build and startup, so a change confined to non-numerical code
clears its records in minutes. A mismatch leaves the record owed and
`scripts/measure_calibration.sh --owed` measures exactly the records that remain.
The rows live in `parity/calibration_probe_attestations.toml`. Each row stores the
dependency fingerprint of the record's test at the probed commit
(`scripts/evidence_fingerprint.py`); `scripts/calibration_status.py` counts the
record as current while a passing row covers it and that fingerprint equals the
tree's. Calibration status never changes support or refuses a route, but CI fails while a record is owed (`scripts/calibration_status.py --check`), so owed records are measured here and committed; see [Earning and checking the records](guides/scientific-verification.md#earning-and-checking-the-records).

## Budget compilation, simulation and collection separately

Batch source changes before a coalesced release prebuild. The embedded coverage
snapshot is several megabytes; changing it invalidates `antecedent-io-core` and
downstream compilation. That dependency explains invalidation, but does not by
itself measure how much compilation time the snapshot causes. Compile time must
be reported separately from numerical wall time. Use a necessary build's Cargo
timings rather than trigger another build solely to time it.

A managed immutable measurement checkout prevents continuing branch work from
changing an active experiment. Give different source checkouts separate build
caches, or reserve a shared cache for the frozen source until its numerical
queue finishes. Limit the *sum* of simulation workers, independent probe and
extension workers, and competing compiler or test workers. A per-process limit
alone does not prevent oversubscription.

When two same-version checkouts share a Cargo target directory, a successful
`--no-run` alone does not prove that the executable matches the second checkout.
If the second checkout's source files carry older timestamps, Cargo can call every
local artifact fresh and reuse the first checkout's binary even though a test
source differs.

The calibration prebuild serializes cooperating builds and fingerprints Cargo
metadata plus crate sources and assets. It advances timestamps only for changed
inputs before invoking Cargo, so older copied timestamps cannot hide different
source bytes. A cache with no source receipt is invalidated once; an unchanged
receipt or Python-only orchestration edit reuses its compiled inputs. The receipt
records each content hash under the target directory. This guard does not replace
exclusive cache ownership when using Cargo directly, and it never removes compiler
artifacts, changes numerical methods, or alters measurement provenance.

Keep the number of build caches bounded. Reuse one target directory for ordinary
branch checks and retain a separate cache only when an active frozen measurement
or probe checkout needs it. Repeated target directories, feature sets and
test profiles can retain many generations of binaries and incremental compiler
state. Share the same profile and feature set where the checks permit it.

Before deleting a cache, check which builds and test executables are using it.
Remove only inactive compiler output; keep raw calibration logs, tally sidecars,
archived measurement binaries and provenance receipts. Do not run `cargo clean`
against an active measurement target. Once an isolated checkout has finished its
measurements and probe obligations, retire its compiler cache. Compare available
disk space before and after cleanup: APFS clones and shared blocks can make the
recovered space much smaller than the sum of directory sizes. For
one-shot debug builds, `CARGO_INCREMENTAL=0` avoids accumulating incremental
state at the cost of slower subsequent compilation; set it before building,
rather than changing the configuration of a running build.

Collection can also become the bottleneck when dependency-graph scans repeat for
each completed group. The wrapper collects once, after its numerical run; raw
logs are durable as soon as each job writes them.

Scheduling and reuse preserve nominal coverage targets, reporting floors,
usefulness limits and minimum scored replicate requirements. The separately
versioned suite-wide coverage decision protocol controls Monte Carlo false
rejections over a source-fixed family and registered looks. It neither turns a
failed legacy gate into a prospective pass nor excuses a scientific floor miss.
A fresh seed domain and frozen source are required for new prospective evidence;
old accepted evidence retains its original source and decision semantics.
