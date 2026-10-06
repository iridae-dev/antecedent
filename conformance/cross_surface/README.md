# Cross-surface fixtures

Artifacts built on one language surface and consumed on the other. Each
consumer builds its expected identity from constants and never from the
producer's bytes or identity files.

## Generation (run from the repository root)

Python-built (`py_*`, consumed by Rust):

    python python/tests/generate_cross_surface_fixtures.py

Rust-built (`rust_*`, consumed by Python):

    ANTECEDENT_WRITE_FIXTURES=1 cargo test -p antecedent-io --test cross_surface_external_claim -- --ignored regenerate_rust_fixtures
    ANTECEDENT_WRITE_FIXTURES=1 cargo test -p antecedent-design --test cross_surface_decision -- --ignored regenerate_rust_fixtures

Consumers: `crates/antecedent-io/tests/cross_surface_external_claim.rs`,
`crates/antecedent-design/tests/cross_surface_decision.rs`,
`python/tests/test_cross_surface.py`. A missing fixture fails its test.

## External response claim (closed form `E[Y | do(a)] = 1 + 2a`)

Coordinates `do(a=0)`, `do(a=1)`, `do(a=2)` of outcome `y` (units `mmHg`,
population `target`, functional `mean`, horizon 0); values `[1, 3, 5]`.
Contract `checked-contract`, graph `graph-1`, provider `lab/curve@v3`, snapshot
`snap-9`, request `req-1`, evidence `factor:z`, assumption `ignorability`,
trust `externally_attested`, support `supported, supported,
outside_empirical_support`, not native estimation.

- `py_external_claim.bin`, `rust_external_claim.bin`: the exported claim.
- `py_external_claim.identity.json`, `rust_external_claim.identity.json`: the
  full claim identity (cross-check only).

## Decision (`P * Q`)

Source rows `p = [1, 3, 2, 0]`, `q = [4, 0, 2, 6]`, `safe = 3` (exact,
joint, interventional-predictive). Contract: `risky` (intervention, `P * Q`)
and `safe` (policy, `max(x0, 0)`), expected utility, hard constraint `q-cap`
(`Q <= 5` with probability at least 0.75, applies to `risky`). Expected:
`EU(risky) = 2`, `EU(safe) = 3`, `EVPI = 0.5`, verdict uniquely optimal `safe`.

- `{py,rust}_decision_source.bin`: the joint distribution artifact.
- `{py,rust}_decision_contract.bin`: the decision contract artifact.
- `{py,rust}_decision_result.bin`: the result, replayable from contract and source.
- `py_decision.identities.json`: Python's contract identity and source digest
  (cross-check only).
