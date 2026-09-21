# Transport identification, catalog, evaluation, and bootstrap

Workloads in `crates/antecedent/benches/transport.rs`:

- `identify_classical_3node` — X→M→Y with X↔Y, selection on Y
- `identify_classical_6node_districts` — two disjoint frontdoor districts
- `catalog_search_1_regime` / `catalog_search_4_regimes` — bounded catalog search after a frozen classical derivation
- `evaluate_exact_3node_8cell` / `evaluate_exact_6node_64cell` — compile+evaluate a bound formula on a uniform joint
- `statistical_plugin_bootstrap_0` / `statistical_plugin_bootstrap_19` — prepared empirical-table re-estimate

Domain limits: classical/catalog sID on binary variables; not meta-transport completeness; not simultaneous bands; bootstrap 19 is a cost probe, not a coverage design. Grid-size cost is the 2-point statistical request vs 19-replicate outer bootstrap. Allocation snapshots for grid prepare/eval remain in `cohesion_allocations`.

Established: 2026-09-21
Machine class: Apple M1 Max (arm64), 64 GB
Criterion sample size: 20
Regression budget: 20%.

## Accepted measurement

| Workload | mean | Gate |
|----------|------|------|
| `identify_classical_3node` | **16.2 µs** | ≤ **19.5 µs** |
| `identify_classical_6node_districts` | **31.7 µs** | ≤ **38.1 µs** |
| `catalog_search_1_regime` | **20.3 µs** | ≤ **24.4 µs** |
| `catalog_search_4_regimes` | **46.0 µs** | ≤ **55.2 µs** |
| `evaluate_exact_3node_8cell` | **28.5 µs** | ≤ **34.2 µs** |
| `evaluate_exact_6node_64cell` | **158 µs** | ≤ **189.6 µs** |
| `statistical_plugin_bootstrap_0` | **4.03 µs** | ≤ **4.84 µs** |
| `statistical_plugin_bootstrap_19` | **157 µs** | ≤ **188.7 µs** |

```bash
cargo +1.85 bench -p antecedent --bench transport
```
