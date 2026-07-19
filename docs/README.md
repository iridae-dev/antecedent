# Documentation

| Doc | Contents |
|-----|----------|
| [Architecture](architecture.md) | Invariants, crates, analysis pipeline, execution model |
| [Development](development.md) | Gates, tests, performance rules, features, versions |
| [Artifacts](artifacts.md) | Wire format, migration, graph interchange |
| [Hot paths](hot_paths.md) | Benches, baselines, allocation contracts |
| [Conformance](conformance/README.md) | Generated from `conformance/` fixtures |
| [Security review](security_review.md) | Unsafe, deps, licensing evidence |

Open work beyond inventories lives in ADRs / parity notes when it reopens.
Decisions: [adr/](../adr/README.md).

Regenerate conformance docs:

```bash
python3 scripts/generate_conformance_docs.py
```

API reference: `cargo doc -p causal --open` / [docs.rs/causal](https://docs.rs/causal).
Python stubs live next to the package (`python/causal/*.pyi`).
