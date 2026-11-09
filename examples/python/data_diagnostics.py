"""Automatic and standalone fit-free data diagnostics.

Preparing a study runs the fit-free diagnostics on its table; the script prints each finding
with its observation and next action, then runs the full suite on all 2,000 deterministic rows
and checks that every row was read. The diagnostics describe the table only: they fit no model
and make no causal claim.

SPDX-License-Identifier: MIT OR Apache-2.0
"""

import antecedent as ant
import numpy as np

rows = {"t": np.arange(2000, dtype=float) % 2, "y": np.arange(2000, dtype=float) % 10}
study = ant.prepare(
    rows,
    graph=[("t", "y")],
    query=ant.AverageEffect("t", "y"),
    estimator="linear.adjustment.ate",
    refute=False,
)
for finding in study.data_diagnostics.findings:
    print(finding.code, dict(finding.observation), finding.next_action)

print(study.data_diagnostics.to_json())
full = ant.diagnose_data(rows, suite=ant.DataDiagnostics(max_sample_rows=None))
assert full.sample_rows == len(rows["t"])
