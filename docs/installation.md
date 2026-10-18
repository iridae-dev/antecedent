# Install and verify Antecedent

Choose a released package for ordinary use or a source checkout when testing an unreleased branch. Read documentation for the same version as your package. This branch is the [2.4.1 release draft](release-notes/v2.4.1.md); the registry installation commands below select the published 2.4 line and do not install an unreleased candidate.

## Python package

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install "antecedent>=2.4.0,<2.5"
python -c "import sys, antecedent; print(sys.executable); print(antecedent.__version__)"
```

The release wheel matrix covers CPython 3.11–3.14 on Linux x86_64 and aarch64 (manylinux), macOS arm64, and Windows x86_64. Other platforms may require a source build. NumPy and Pydantic are installed as dependencies. The default numerical backend needs no system BLAS installation.

For notebooks, install `pandas matplotlib ipykernel jupyterlab` in that environment, run `python -m ipykernel install --user --name antecedent-24`, and select that kernel. After upgrading an imported native extension, restart the kernel. [Choose an example](examples.md) for its additional dependencies.

## Testing 2.4.1 before publication

Installing from PyPI does not install an unreleased branch. In a checkout of the `2.4.1` branch, create and activate a virtual environment, then:

```bash
python -m pip install maturin
maturin develop --release --manifest-path python/Cargo.toml
python -c "import antecedent; print(antecedent.__version__, antecedent.__file__)"
python examples/python/notebook_to_system_24.py
```

Rust 1.85 is selected by `rust-toolchain.toml`; install it through rustup if needed. Use `--release` for representative estimation performance. Python wheels enable the tree and neural learner backends. See [development](development.md) for contributors' checks and supported build combinations.

## Rust package

In your application's directory:

```bash
cargo add antecedent@2.4.0
cargo doc -p antecedent --open
```

The facade's default feature set is empty. Enable `ml-gbdt`, `ml-forest`, `ml-full` (both tree backends), or `ml-neural` when the corresponding learner is needed. For example, `cargo add antecedent@2.4.0 --features ml-forest`. Optional features enable backends; they do not grant new scientific claims.

From the branch checkout, use [the Rust quickstart](rust-quickstart.md). It runs against the source you checked out.

## Resolve installation problems

| Symptom | Action |
| --- | --- |
| Notebook imports an older version | Print `sys.executable` and `antecedent.__file__`, install into that interpreter, then restart the kernel. |
| pip starts compiling Rust | Check Python/platform wheel availability and update pip. A source build requires the pinned Rust toolchain. |
| Optional handoff says it was skipped | Install `dowhy` or `econml` in the same environment before running that integration. |
| Example and API disagree | Compare package version, checkout ref, and documentation version before changing the analysis. |
