# Antecedent

Explicit causal inference for **Python** and **Rust**.

| Doc | Contents |
|-----|----------|
| [Architecture](architecture.md) | Invariants, crates, analysis pipeline, execution model |
| [Development](development.md) | CI vs local gates, tests, performance rules, versions |
| [Artifacts](artifacts.md) | Wire format, migration, graph interchange |
| [Prior bank](prior_bank.md) | External prior catalog, compose, conflict, transport |
| [API naming](api_naming.md) | Rust ↔ Python capability dictionary |
| [Hot paths](hot_paths.md) | Benches, baselines, allocation contracts |
| [Conformance](conformance/README.md) | Generated from `conformance/` fixtures |
| [Security review](security_review.md) | Unsafe, deps, licensing evidence |

## API reference

- **Python:** [Python API](python-api.md) on this site (`/python/` via pdoc; no download)
- **Rust:** [docs.rs/antecedent](https://docs.rs/antecedent); locally `cargo doc -p antecedent --open`

Decisions: see `adr/` in the repository.

Regenerate conformance pages:

```bash
python3 scripts/generate_conformance_docs.py
```
