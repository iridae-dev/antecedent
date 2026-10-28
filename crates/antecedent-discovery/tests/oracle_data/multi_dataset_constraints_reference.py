# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Link assumptions of the pooled multi-dataset search for every pair of node roles and lag.

Regenerate with:  uv run --script multi_dataset_constraints_reference.py > multi_dataset_constraints_reference.json

The five roles are a system variable, an observed space context, an observed time context, the
synthetic space dummy and the synthetic time dummy.  A directed link source(lag) -> target(0) is
forbidden when its target is any context or dummy (nothing causes an exogenous node), and when
its source is an exogenous node other than the time context and the link is lagged (the space
context, both dummies are constant within an environment or time-index stand-ins, so only the
time context may act with a delay).  Every other pair is allowed: system to system at any lag,
and context or dummy to system, lagged only for the time context.  The table is written out for
lags 0 to 3 and is the executed statement of the rule list recorded in the module; it is not an
independent derivation from the source paper, which was not retrieved.
Repository-authored (MIT OR Apache-2.0).
"""
import itertools
import json

ROLES = ["system", "space_context", "time_context", "space_dummy", "time_dummy"]
EXOGENOUS = set(ROLES[1:])
rows = []
for src, tgt, lag in itertools.product(ROLES, ROLES, range(4)):
    forbidden = tgt in EXOGENOUS or (src in EXOGENOUS and lag > 0 and src != "time_context")
    rows.append({"source": src, "target": tgt, "lag": lag, "forbidden": forbidden})
print(json.dumps(rows))
