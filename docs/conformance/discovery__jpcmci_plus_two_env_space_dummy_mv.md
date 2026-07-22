# J-PCMCI+ multivariate space-dummy pin

**Suite path:** `conformance/discovery/jpcmci_plus_two_env_space_dummy_mv`

Exercises `SpaceDummyCiMode::MultivariateBlock` with `include_space_dummy=true`
on ≥3 environments (so ≥2 one-hot space columns collapse to one logical node).

Library-only expected (no external multivariate space-dummy oracle required).

## Expected summary

Top-level keys: `algorithm_id, include_space_dummy, max_logical_space_dummy_ids, min_envs, min_links, min_nodes, notes, require_multivariate_diagnostic, space_dummy_ci, tolerance_class` (10 fields).
