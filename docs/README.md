# Documentation

| Doc | Contents |
|-----|----------|
| [Architecture](architecture.md) | Invariants, crates, analysis pipeline, execution model |
| [Development](development.md) | Gates, tests, performance rules, features, versions |
| [Artifacts](artifacts.md) | Wire format, migration, graph interchange |
| [Prior bank](prior_bank.md) | External prior catalog, compose, conflict, transport |
| [API naming](api_naming.md) | Rust ↔ Python capability dictionary |
| [Hot paths](hot_paths.md) | Benches, baselines, allocation contracts |
| [Conformance](conformance/README.md) | Generated from `conformance/` fixtures |
| [Security review](security_review.md) | Unsafe, deps, licensing evidence |

Decisions: [adr/](../adr/README.md).

Regenerate conformance docs:

```bash
python3 scripts/generate_conformance_docs.py
```

API reference (private repo): download `docs.tar.gz` from a GitHub Release, or
build locally with `cargo doc -p causal --open`. Python API HTML is generated
with pdoc in CI (`docs.yml` / release workflow). docs.rs is not used while
crates stay unpublished.
Rust ↔ Python names: [api_naming.md](api_naming.md).
Python stubs live next to the package (`python/causal/*.pyi`).
