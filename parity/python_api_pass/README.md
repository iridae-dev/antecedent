# Python API pass: capability and journey inventory

This is an **open implementation inventory**, not a parity or release claim.
The approved design is a clean Python API with functions and typed objects,
immutable branches, explicit execution, shared outcome/refusal inspection, one
owned data binding, direct composition, and typed artifacts. Integration owns
all native builds and executing family runs.

`capabilities.toml` reconciles every existing surface-registry row, including
all 120 `internal` classifications. None inherits approval from a resolving
name or a nonempty reason. Each old row has an exact Rust symbol, named owner,
behavior, and explicit open action until consuming assertion evidence is reviewed.
There are also independently named missing method/lifecycle capabilities and
one narrowly mapped actual temporal-ADMG pulse journey. That mapping says what
its assertions establish; it does not grant missing lifecycle/artifact coverage.

`rust_surface.json` preserves the committed facade snapshot's methods,
declarations, free functions, and reexport edges with exact owners/signatures
and alias targets. It additionally records public source-module declarations
omitted by normal-feature rustdoc, including calibration-internal entries.
Unmapped entries are closure gaps. Many represent aliases, DTOs, or related
methods rather than independent missing bindings; reconcile them to reviewed
capabilities and actual consumers instead of introducing blanket alias or
plumbing exclusions. The snapshot is not the public API of every lower crate;
the existing API generation gate remains authoritative for source freshness.

## Meaning of evidence

A capability's `required_dimensions` describes its actual contract. Use
`success`, `inspection`, `controls`, `refusal`, `branch`, `lifecycle`, and
`artifact` only when relevant. Evidence names an exact module-level test,
reachable same-module assertion scopes, and a reviewed semantic assertion
contract. A helper must actually be called from that test. Static AST inspection
checks the reference is meaningful; it cannot prove the selected path executed
or that an assertion proves the stated science. Reviewer examination and the
executing runner are separate requirements.

`status = "plumbing"` additionally requires an exact `composed_caller` chain
and executing assertions on that consumer. A Python-authored declaration of an
observed effect is not proof of Rust-derived invalidation. `status = "gap"`
remains open regardless of the old registry status. `covered` means reviewed
mapping only until matching execution evidence exists. Skip, xfail, no-assertion,
empty selection, failed call/teardown, stale source, and replaced native binaries
cannot establish completion.

## Commands

- `python3 scripts/check_python_capabilities.py` checks inventory integrity and
  reports open gaps. Its success **does not** mean the API pass is complete.
- `python3 scripts/test_check_python_capabilities.py` runs dependency-free
  negative controls without building or importing the extension.
- Integration executes `python scripts/run_python_capabilities.py
  --build-receipt /absolute/build-receipt.json --output /absolute/execution.json`
  only after its serialized native build, using the intended Python environment.
- `python scripts/check_python_capabilities.py --require-complete
  --execution /absolute/execution.json` is the closure gate. It refuses every
  unresolved mapping/dimension and requires actual passing execution of each
  selected test, including all collected parameters.

Integration's build receipt must be captured after the actual source-stable
build, binding `source_digest` from `check_python_capabilities.source_digest`,
absolute `native_path`, SHA-256 `native_sha256`, Python major/minor array `python`,
and the actual `build_command`. It is an integration provenance record, not a
cryptographic proof that arbitrary binary bytes implement source. The runner
verifies the loaded extension and receipt, hashes source again after execution,
records selected/collected/passed nodeids and refuses changed source/binaries.
It disables inherited pytest selection options and does not build or install.
Do not manufacture a fresh receipt around an old extension.

The older `check_python_surface.py`, registry, product matrix, acceptance
manifest, frozen namespace and native stub checks retain their separate roles.
The new files do not silently replace those checks or bless their exclusions.
