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

## Composition bundle (C3)

One bundle with two hand-derived decisions, nodes `contract`, `law`, `result`,
`claim`, `mean_contract`, `mean_result` and edges `law -> result`,
`contract -> result`, `claim -> mean_result`, `mean_contract -> mean_result`.

- Joint law: the `P * Q` source and contract above; `result` reads
  `risky.expected_utility = 2` and `safe.expected_utility = 3`.
- Point only: the external claim above (`E[Y | do(a)] = 1 + 2a`, values `[1, 3, 5]`)
  with utility `2 * mean - 1`; `mean_result` reads `wait.expected_utility = 1`
  (`a = 0`) and `treat.expected_utility = 9` (`a = 2`). The claim is `mean_only` and
  `externally_attested`, so the bundle label is `point_only_attested`.

Files: `py_composition_bundle.bin` and `py_composition_bundle.identity.json`
(bundle identity, cross-check only), `rust_composition_bundle.bin`.

Generation (from the repository root):

    python python/tests/generate_cross_surface_bundle_fixtures.py
    ANTECEDENT_WRITE_FIXTURES=1 cargo test -p antecedent-design --test cross_surface_bundle -- --ignored regenerate_rust_fixtures

Consumers: `crates/antecedent-design/tests/cross_surface_bundle.rs` (builds the
expected bundle from Rust constants and consumes `py_composition_bundle.bin` under its
identity) and `python/tests/test_cross_surface_bundle.py` (builds the expected bundle
in-process and consumes `rust_composition_bundle.bin`). A missing `py_*` fixture fails
the Rust test; a missing `rust_*` fixture fails the Python test.
