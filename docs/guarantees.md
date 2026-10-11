# What “supported” means

Three propositions must remain separate.

| Term | What it means | What it does not mean |
| --- | --- | --- |
| **Capability** | An implementation or type exists. | Every combination using it is permitted. |
| **Licensed support** | A specific query × graph × structure × inference × validation route has a recorded execution/evidence contract. | The real-world causal assumptions are true. |
| **Scientific validity** | The assumptions are appropriate for the real situation. | Something software can establish from a successful call alone. |

The [support matrix](support-matrix.md) is the public license for ordinary
analysis combinations. Specialized [measured scalar adapters](population-time-uncertainty.md)
and [decision workflows](lifecycle.md) have separate checked contracts,
without expanding that grid. [Capabilities](capabilities.md) explains what exists. A successful run
still requires a defensible causal model, relevant data, and interpretation of
the result's support, uncertainty, and assumptions.

Calibration has the same discipline: a reported interval is `calibrated` only
when a matching coverage record still attests the current code and the
execution lies inside that record's measured scope. The release attestation
gate checks that every published record still matches this build, including
any scoped replay waiver for a change that cannot move its measured number.

Licensed does not mean measured. The [support matrix](support-matrix.md)
states, for each licensed cell, whether a coverage record is cited, whether
the estimator has no coverage measurement (`estimator_grid_not_measured`),
or whether the cell reports no interval.
