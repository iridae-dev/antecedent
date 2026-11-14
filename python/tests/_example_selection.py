"""Which shipped examples and notebooks the Python suite runs, and what each run needs.

`test_examples_run.py` parametrizes over these, and `scripts/example_coverage.py` reads the
same values to account for every catalog entry, so the report cannot drift from the suite.
Nothing here imports pytest or the package: it is data about selection only.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

EXAMPLES = sorted((_ROOT / "examples" / "python").glob("*.py"))
NOTEBOOKS = sorted((_ROOT / "examples" / "notebooks").glob("*.ipynb"))

# An example that cannot run without an optional module is skipped, not failed, when the
# module is absent: the protocol script fits with SciPy, the Parquet example writes its
# input with PyArrow (without it the example only prints its skip), and three walkthroughs
# render figures.
EXAMPLE_REQUIRES = {
    "cyclic_paper_experiments.py": "scipy",
    "parquet_e2e.py": "pyarrow",
    "marketing_channel_structural_uncertainty.py": "matplotlib",
    "marketing_experiment_design.py": "matplotlib",
    "sales_campaign_prior_transfer.py": "matplotlib",
}

# Every notebook loads a table and draws a figure.
NOTEBOOK_REQUIRES = ("pandas", "matplotlib")
