# J-PCMCI+ two-env edge-set pin

**Suite path:** `conformance/discovery/jpcmci_plus_two_env_edges`

System-only J-PCMCI+ edge-set vs pinned baseline (`include_space_dummy=false`).
Oracle regeneration: see `expected.json` → `reference.command` (out-of-repo).
Requires a baseline build that ships J-PCMCI+ (pin ≥5.2.9 line).
Space-dummy multivariate CI parity is deferred, not this fixture.

## Expected summary

Top-level keys: `algorithm_id, alpha, fdr, include_space_dummy, max_lag, min_lag, n_envs, n_per_env, notes, reference, tolerance_class, true_parents, var_names` (13 fields).
