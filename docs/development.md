# Development

## Gates

Feature gates own inventory honesty + conformance for a domain:

```bash
bash scripts/gate_estimate_ci.sh
bash scripts/gate_bayesian.sh
bash scripts/gate_gcm.sh
bash scripts/gate_pag.sh
bash scripts/gate_context.sh
bash scripts/gate_attribution.sh
bash scripts/gate_design_state.sh
bash scripts/gate_upstream_names.sh
bash scripts/gate_calibration.sh   # SE coverage / CI Type I — release / weekly, not every PR
bash scripts/gate_release.sh
```

Mark a `parity/*.toml` capability `done` only with conformance under `conformance/`
**or** a named harness in the gate script, plus a recorded reference-generation
command when black-box comparison applies.

Statuses: `pending` | `in_progress` | `done`. No waiver vocabulary.

## Tests that matter

| Kind | Role |
|------|------|
| Unit / property | Algorithm invariants, graph witnesses, numeric edge cases |
| Conformance | Frozen fixtures vs expected outputs (`conformance/`) |
| Calibration | Coverage / Type I / null FPR (`gate_calibration.sh`) |
| Cross-language | Python bindings exercise the same semantics |
| Criterion benches | Designated hot paths; regressions beyond budget block merge |
| Fuzz | Parsers / graph / artifact surfaces under `fuzz/` |

Tolerance classes live in `causal-core` (ADR 0010). Do not tighten or loosen a
conformance band without an ADR-level reason.

## Performance rules (merge blockers)

- Data layout and copy policy are designed with the algorithm, not after.
- No per-observation dynamic dispatch / Python / hash / heap in scalar inner loops
  unless the slow path is API-explicit and separately benched.
- Scalar kernels are the correctness reference; SIMD/BLAS/parallel paths pass the
  same tests.
- Do not change statistical semantics to go faster (sample selection, masking,
  conditioning order, randomization, stopping rules, estimands).
- Parallelism is bounded by `ExecutionContext`.
- Superlinear storage must expose bounds, streaming, or refuse — not OOM later.

See [hot_paths.md](hot_paths.md).

## Feature flags

Cargo features mean “optional adapter / heavy backend,” never “different numbers
on the default path.”

Always on: `faer`, portable kernels, `ExecutionContext` parallelism (`rayon`
rejected).

Present today (examples): `causal-data/arrow`, `causal-model/gaussian-process`,
`causal-prob/hmc`. Reserved / unfinished: `smc`, `simd-runtime`. Optional ingest
and exchange adapters may land later without reshaping core types.

## Unsafe / deps

Reviewed `unsafe` is concentrated in `causal-kernels` (SIMD) and thin IO mmap.
New `unsafe` needs justification in review. Dependency and license policy:
[security_review.md](security_review.md), ADR 0008.

## Versions

Packages stay at **0.1.0** until an explicit 1.0 decision (ADR 0017). Artifact
format is frozen separately — see [artifacts.md](artifacts.md).

MSRV: Rust 1.85, edition 2024. Python: CPython 3.11–3.14.
