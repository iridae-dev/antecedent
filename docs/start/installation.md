# Install Antecedent

Install a package for ordinary use, or build a source checkout for development. Python needs 3.11 or later; Rust needs 1.85 or later.

## Python package

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade antecedent
python -c "import sys, antecedent; print(sys.executable); print(antecedent.__version__)"
```

The wheel matrix covers CPython 3.11–3.14 on Linux x86_64 and aarch64 (manylinux), macOS arm64, and Windows x86_64. Other platforms may require a source build. NumPy and Pydantic are installed as dependencies. The default numerical backend needs no system BLAS installation. The wheel includes the CPU-native neural nuisance learner; no separate extra is needed.

To keep an environment on the 3.0 line, install `"antecedent>=3.0"`.

For notebooks, install `pandas matplotlib ipykernel jupyterlab` in that environment, run `python -m ipykernel install --user --name antecedent`, and select that kernel. After upgrading an imported native extension, restart the kernel. [Choose an example](../examples.md) for its additional dependencies.

## Rust package

In your application's directory:

```bash
cargo add antecedent
cargo doc -p antecedent --open
```

The facade's default feature set is empty. Enable `ml-gbdt`, `ml-forest`, `ml-full` (both tree backends), or `ml-neural` when the corresponding learner is needed. For example, `cargo add antecedent --features ml-forest`. Optional features enable backends; they do not grant new scientific claims. See the [capability inventory](../capabilities.md) for the learners each feature provides.

The [published Rust reference](https://docs.rs/antecedent/) describes the published crate. `cargo doc` on a checkout is the reference for the source you are running.

## Build from source

In a checkout of the version you want to build, create and activate a virtual environment, then:

```bash
python -m pip install maturin
maturin develop --release --manifest-path python/Cargo.toml
python -c "import antecedent; print(antecedent.__version__, antecedent.__file__)"
python examples/python/notebook_to_system.py
```

Rust 1.85 is selected by `rust-toolchain.toml`; install it through rustup if needed. Use `--release` for representative estimation performance. Python wheels enable the tree and neural learner backends. See [development](../development.md) for contributors' checks and supported build combinations.

## Check the installation

Run [your first analysis](first-analysis.md). It prepares, inspects and estimates an effect in simulated data with a known answer.

## Resolve installation problems

| Symptom | Action |
| --- | --- |
| Notebook imports an older version | Print `sys.executable` and `antecedent.__file__`, install into that interpreter, then restart the kernel. |
| pip starts compiling Rust | Check Python and platform wheel availability and update pip. A source build requires the pinned Rust toolchain. |
| Optional handoff says it was skipped | Install `dowhy` or `econml` in the same environment before running that integration. |
| Example and API disagree | Compare package version, checkout ref and documentation version before changing the analysis. |
