# Acceptance manifest

`parity/acceptance_manifest.toml` freezes, before implementation, what a practitioner must be able to do and what the library must refuse. Each `[[case]]` is one executable acceptance case: a public route, a concrete supported configuration, the required outputs, frozen precision, power, reporting, coverage, runtime and memory targets, and a declarative description of the simulated dataset it runs on. The authors' original datasets are not available, so every dataset is a simulated representative of the described study, generated from the stated generator and seed.

Cases are grouped by `area` (ingest, identifiers, missing data, adaptive procedures, discovery, calibration ranges, practitioner scale, graph ceilings, transport, release closure and so on). Cases are named for what they show, never for a release or planning label.

## Schema

| Field | Meaning |
| --- | --- |
| `id` | Unique scientific snake_case name. |
| `area` | One of the areas listed in `scripts/check_acceptance_manifest.py`; every area has at least one case. |
| `workflow` | The public Python call or Rust facade function a practitioner uses. |
| `configuration` | Inline table of the supported configuration: sizes, arms, responses, variables, steps, draws, candidates, studies. |
| `expected` | `kind` is `value`, `refusal`, `interval`, `calibrated_interval` or `receipt`, with the required `outputs`. A refusal also states `refusal`. |
| `targets` | `runtime_seconds` and `memory_bytes` on the reference workstation (8 cores, 16 GiB), plus the rates the kind needs, and a `rationale`. |
| `dataset` | `generator`, `seed`, `n` and the effect sizes, correlation or priors, with `simulated_representative = true`. |
| `evidence` | `status = "planned"` until a test binds it; then `status = "bound"` and `test = "<path>::<fn>"`. |

Interval cases state `nominal_coverage` and a `coverage_floor` below it. A `calibrated_interval` case also requires at least 2,000 replicates, a reporting rate of at least 0.90 and a precision target (`max_median_*`).

Targets are frozen. Later work may bind evidence but may not loosen a target without a recorded reason in the change that does so.

## Run the checker

```bash
python3 scripts/check_acceptance_manifest.py
python3 scripts/check_acceptance_manifest.py --require-bound
python3 scripts/check_acceptance_manifest.py --self-test
```

The default run validates the schema, unique ids, area coverage, and, for every `bound` case, that the test file exists and defines the named function (`fn name` in Rust, `def name` in Python). `--require-bound` additionally fails on any `planned` case and is used at release closure. `--self-test` exercises the checker on in-memory manifests.
