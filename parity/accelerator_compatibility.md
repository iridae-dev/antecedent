# Accelerator compatibility (G07)

Checked 2026-10-19. Compatibility memo only. No backend is locked. No timing was taken. Absence of a GPU path is not a negative GPU result.

`parity/accelerator_decision.md` still stands: the CPU wheel is the available backend, and `ml-neural` is Burn `NdArray<f32>` only.

## In-repo pins

| Pin | Declared | Resolved (`Cargo.lock`) | Where |
| --- | --- | --- | --- |
| Rust | 1.85, edition 2024 | toolchain channel `1.85` | `rust-toolchain.toml`; `Cargo.toml` `workspace.package.rust-version`; ADR 0007 |
| Arrow | `arrow-array` / `arrow-schema` / `arrow-buffer` `54` | `54.3.1` | `Cargo.toml` workspace dependencies |
| PyO3 | `0.29` (`pyo3-build-config` `0.29`) | `pyo3` `0.29.2` | `Cargo.toml`; `python/Cargo.toml` uses the workspace pin. ADR 0007: CPython 3.11–3.14 |
| Burn | `0.16`, features `std`, `autodiff`, `ndarray` only | `0.16.1` | `Cargo.toml`; `crates/antecedent-learn-burn` |
| faer | `0.24` | `0.24.4` | `Cargo.toml`. ADR 0023: faer stays the statistical linear-algebra backend |

The Burn adapter is `Autodiff<NdArray<f32>>` / `NdArray<f32>` (`crates/antecedent-learn-burn/src/lib.rs`). No `wgpu`, `metal`, `cuda`, or `cuda-jit` feature is enabled. ADR 0023: Burn/CUDA/WGPU are not foundational; a later GPU provider must still emit `Vec<f64>`.

Python default features include `ml-neural` (`python/Cargo.toml`). That flag selects the CPU adapter. It does not select a device.

## Apple Silicon candidate (Metal / Burn WGPU)

Separate from NVIDIA. This is the neural-learner lane, not cuDF.

Observed:

- Pinned Burn `0.16.1` declares `rust-version = "1.85"` and optional features `wgpu` and `metal` (not enabled here). Source: `https://raw.githubusercontent.com/tracel-ai/burn/v0.16.1/crates/burn/Cargo.toml` (fetched 2026-10-19).
- Burn WGPU 0.16 README: Metal is GPU-only and listed for macOS and iOS, not Linux. Vulkan/OpenGL/WebGPU are other WGPU targets. Source: `https://raw.githubusercontent.com/tracel-ai/burn/v0.16.1/crates/burn-wgpu/README.md`.
- No separate NVIDIA-style driver pin applies. Metal is the OS GPU API. A driver version was not looked up and is not claimed.
- Current stable Burn is `0.22.0` (published 2026-10-06), `rust_version` `1.95`, with optional features `wgpu`, `metal` (enables `wgpu`), and `cuda`. Source: `https://crates.io/api/v1/crates/burn`. Last Burn release whose crates.io `rust_version` is `1.85` is `0.18.0` (2025-07-18). `0.19.1` is already `1.89`.

Opt-in without touching CPU wheels is possible only as a non-default feature or a separate distribution. The published CPU build must keep today's features (`std`, `autodiff`, `ndarray`). `python` defaults must not gain `wgpu` or `metal`. Cargo features are compile-time: a CPU wheel built without them does not link the WGPU/Metal backend. That split does not exist in-tree yet.

Moving the workspace pin to Burn `0.22.0` to obtain the current Metal/WGPU stack requires Rust `1.95`, which contradicts the 1.85 toolchain and ADR 0007.

## NVIDIA candidate (CUDA / cuDF / Burn CUDA)

Separate from Apple Metal. Two different NVIDIA stacks exist; neither is the Rust Polars engine.

**Python cuDF backend.** Polars GPU support is a Python Lazy API engine (`engine="gpu"`) on NVIDIA GPUs via RAPIDS cuDF. Unsupported ops can fall back to CPU. Results collect as CPU-backed frames. Source: `https://docs.pola.rs/user-guide/gpu-support/` (fetched 2026-10-19).

RAPIDS 26.10 platform support (`https://docs.rapids.ai/platform-support/`):

- Linux, glibc 2.28+, x86_64 and aarch64; Windows only via WSL.
- CUDA 12: toolkit 12.2–12.9, driver 535+.
- CUDA 13: toolkit 13.0–13.3, driver 580+.
- Compute capability 7.0+ (Volta+) on CUDA 12; Turing+ on CUDA 13.

The RAPIDS install guide states different driver floors (CUDA 12 driver `525.60.13+`, CUDA 13 driver `580.65.06+`). The two NVIDIA pages do not agree on a single driver number. Do not lock a driver from either page alone.

PyPI metadata, fetched 2026-10-19:

- `cudf-polars-cu12` `26.10.0` (uploaded 2026-10-08): `polars>=1.35,<1.45`, `cuda-bindings>=12.9.2,<13.0`, `requires_python >=3.11`. `https://pypi.org/pypi/cudf-polars-cu12/json`
- `cudf-polars-cu13` `26.10.0`: same Polars bound, `cuda-bindings>=13.0.1,<14`. `https://pypi.org/pypi/cudf-polars-cu13/json`
- Python `polars` latest: `2.0.0`, `requires_python >=3.10`. `https://pypi.org/pypi/polars/json`

Polars `2.0.0` is outside `polars>=1.35,<1.45`. The 14 October note is unchanged on 19 October.

**Rust Polars.** The Rust Polars lazy engine is not the Python cuDF GPU backend. crates.io `polars` `0.55.2` (2026-08-06) publishes no `rust_version`. Tag `rs-0.55.2` `rust-toolchain.toml` is `nightly-2026-04-01`. Its `Cargo.toml` has 155 features and none named gpu, cuda, cudf, metal, or wgpu. Its `arrow` dependency is package `polars-arrow`, not arrow-rs `54`. Sources: `https://crates.io/api/v1/crates/polars`, `https://raw.githubusercontent.com/pola-rs/polars/rs-0.55.2/rust-toolchain.toml`, `https://raw.githubusercontent.com/pola-rs/polars/rs-0.55.2/crates/polars/Cargo.toml`, workspace `Cargo.toml` on that tag.

**Burn CUDA.** On the pinned crate this is the optional `cuda-jit` feature of Burn `0.16.1`, not cuDF and not enabled. On Burn `0.22.0` the feature is `cuda`, and the crate MSRV is `1.95`. A CUDA toolkit/driver would apply only to a wheel built with that feature. It does not run on Apple Metal.

## Conflicts that forbid a lock

1. Apple Metal/WGPU and NVIDIA CUDA/cuDF are different platforms, drivers, and code paths. One lock cannot cover both. The development host in `parity/accelerator_decision.md` is Apple Silicon; cuDF's documented OS is Linux or WSL2.
2. Workspace Rust is 1.85 (ADR 0007). Current Burn with a maintained GPU feature set is `0.22.0` / Rust `1.95`. Pinning that crate is an MSRV break, not an opt-in flag.
3. `cudf-polars-cu12==26.10.0` and `cudf-polars-cu13==26.10.0` require `polars>=1.35,<1.45`. Current Python Polars is `2.0.0`. No compatible pair was published under those names as of this check.
4. Rust Polars `0.55.2` is not Python Polars `2.0.0`, has no GPU feature, uses `polars-arrow` rather than arrow-rs `54.3.1`, and its upstream toolchain is nightly. It cannot be treated as the cuDF engine or as a drop-in beside the current Arrow pin. Interoperability was not built (no cargo run).
5. ADR 0023 keeps faer as the linear-algebra backend and requires GPU predictions to re-enter as `Vec<f64>`. Nothing in-tree implements that device path (`accelerator_decision.md`).

## Recommendation

Do not lock an accelerator.

Before G02 (roadmap G2: an opt-in candidate that can be measured, not a promotion), all of the following have to be true:

- The candidate is one platform. The other platform stays out of that build.
- CPU wheels and default `ml-neural` stay on Burn `0.16` `NdArray<f32>`. The device feature is off that default and off the CPU wheel.
- A Burn device candidate on this toolchain uses the existing `0.16.1` optional feature (`wgpu`/`metal`, or `cuda-jit`), whose `rust-version` is 1.85. Adopting Burn `0.22.0` waits on an ADR 0007 amendment to Rust 1.95.
- A Python cuDF candidate waits on a `cudf-polars` release whose `requires_dist` accepts the Polars version in that separate environment. `26.10.0` does not accept Polars `2.0.0`. That environment is not the Rust Polars crate.
- G02 still produces no speedup claim. Measurement, if any, comes later and includes transfer.
