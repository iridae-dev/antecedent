# Focused performance investigation

These probes diagnose N20 results. They do not complete its full control,
hardware, row-capacity or exact-capacity grid. Generated measurements, timing
dumps and local reports belong under `benchmarks/performance/investigations/`,
which is ignored by Git; retain the reusable harness and tests in the repository.

First check that the intended Python environment imports the checkout:

```sh
python/.venv/bin/python -c 'import antecedent, antecedent._native; print(antecedent.__file__); print(antecedent._native.__file__)'
```

For an editable checkout, rebuild/install the matching optimized extension:

```sh
env -u CONDA_PREFIX UV_CACHE_DIR=/tmp/antecedent-uv-cache VIRTUAL_ENV="$PWD/python/.venv" python/.venv/bin/maturin develop --release --manifest-path python/Cargo.toml --no-default-features -j 4
```

When repairing an already installed editable environment with its dependencies in
place, add `--skip-install` to rebuild the extension in place. This avoids invoking
the package installer; verify imports and run the workflow tests afterward.

The Python probe includes complete bootstrap or Gaussian posterior uncertainty.
It fails on import errors, cancellation, early stopping, incomplete replicate/draw
counts or failed known-truth checks. One declared warmup precedes each cell;
every measured repeat, native stage time, effort count and extension hash is
retained. A `prepared_estimate` is the existing public operation, with its actual
work measured; it is not assumed to be a zero-refit lookup. Unavailable native
counters are listed explicitly rather than supplied as invented zeros.

```sh
python/.venv/bin/python benchmarks/performance/python_workflows.py --rows 10000 --threads 1 --repeats 7 --bootstrap 199 --draws 400 --output /tmp/python-n10000-t1.json
```

Rust first-answer benchmark IDs bind row and thread counts. Its existing control
uses 199 bootstrap replicates and 400 posterior draws at rows >= 10000.

```sh
cargo bench -p antecedent --bench user_path --no-run
USER_PATH_N=10000 USER_PATH_THREADS=1 target/release/deps/user_path-<hash> --bench user_path_dag_ate_bayesian_laplace --sample-size 20 --warm-up-time 0.5 --measurement-time 2
```

Compile before measuring; run the built executable directly when collecting
process RSS. `time cargo bench` can report the compiler's peak. Keep thread,
row, uncertainty, seed and fixture settings matched across versions. A 10000-row
run and a 100000-row run are different cells, not a regression comparison.

The discovery benchmarks explicitly fund the requested admission-pool workers.
PCMCI checks complete serial/parallel scientific output agreement before timing.
`ci_search` retains all five wall-clock samples and reports their median.

Gaussian prepared ATEs retain an immutable design and its Gaussian sufficient
statistics, not a posterior result. Every call resolves the current prior and
regenerates its requested draws. Changed values, validity, masks, weights or
schema invalidate the source binding; changed inference configurations start a
new workspace. `exec.design.cached` makes a reused design observable. This
retention charges buffer capacities and metadata against the cache byte cap,
including Python's default 4 MB cap, with one bounded-shape entry per handle.
Finite execution-memory/admission policies use the ordinary path. It does not claim reuse across new-data refreshes or arbitrary
retargets. Linear prepared calls also keep the checked design for the identical
immutable table binding. OLS bootstrap resamples use coefficient-only pivoted QR;
the resampling law, replicate count and rank checks are unchanged.
