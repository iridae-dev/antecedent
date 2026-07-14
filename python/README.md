# Python package for causal-library

Requires CPython 3.11+ and a Rust 1.85 toolchain.

```bash
cd python
uv venv && source .venv/bin/activate
uv pip install maturin numpy
maturin develop
python -c "import numpy as np; import causal; print(causal.load_float64_columns(['x'], [np.array([1.,2.,3.])]).bytes_copied)"
```

The initial surface only loads float64 columns and reports copy diagnostics.
