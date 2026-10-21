# Shared execution accounts and native observations in 3.0

Batch workers now use the shared indexed scheduler and bounded inner contexts.
Ordinary and prepared batches use `try_map_indexed`, which admits serial jobs as
well as parallel workers and preserves typed admission failures. Nested work reuses
the live enclosing CPU reservation instead of acquiring a second set of slots.
Cancellation is checked before work starts. Context-free component maps retain
their original compatibility behavior.
Numerical scratch uses the enclosing workflow admission account, including worker
execution. Standalone numerical components reserve their scratch bytes separately.
Admission peaks describe reservations, not process resident memory.

Cancellation wakes admission waiters and refuses new work, but no longer returns
reservations held by live workers or retained buffers. Reservations return on
explicit release or owner drop, after cleanup. This prevents a concurrent job
from reusing RAM or CPU slots while the cancelled job still owns them. A refused
child request after cancellation leaves the parent's cleanup responsibility
intact. Callers must release their buffers and leases rather than relying on a
budget observation to reclaim cancelled work.

`Study::run_observed` and `PreparedStudy::estimate_retained_observed` retain work
reports and execution facts on success, failure and cancellation. Static results
expose observations under `performance.native_execution`, also available in Python.
Enclosing model fits are counted separately from their internal numerical solves.
Optional `observed_*` fields are absent where the corresponding boundary was not
measured. Instrumentation coverage remains incomplete across the full library.

Composite result and performance wires emit an optional `native_execution_v2`
record. It adds optional copied buffer bytes, retained admitted output reservation
bytes, and an explicit
retention mode. V1 records remain readable, but their omitted buffer volumes and
retention mode are unknown; their coverage bits cannot manufacture missing values.
Earlier artifacts without observations remain readable. Unknown observation
formats are refused, including by older readers that only support V1. Imported
observations remain historical records and never become a fresh native execution.
They grant no scientific or execution authority. Licensing and calibration are unchanged.

`OutputRetention::Unmeasured` replaces inferred summary-only mode when no output
handoff was observed. It grants no capacity reuse. Buffer retention volume counts
instrumented admitted output reservations cumulatively, including conservative
capacity bounds held by actual leases. It is distinct from exact heap allocation,
peak admitted memory, process RSS, and caller declarations. A run with multiple output storage modes
reports an unmeasured aggregate mode rather than choosing one arbitrarily.

The Bayesian dose-grid facade exposes `evaluate_observed`, `estimate_observed`,
and `consume_observed`. Python dose-grid posterior and cyclic results expose
`execution_facts`; native refusal and cancellation exceptions retain the same
record on `error.execution_facts`, including failed independent artifact consumption.
These facts describe the attempt and do not make a failed answer publishable.
Cyclic NumPy output views retain admission in their actual native base owner,
so discarding receipts or prepared handles cannot release a live array's budget.

This is the resource and observation foundation for the compiler integration
repair. It does not close H03/H04 or establish complete workflow resource accounting.

Execution observations now include retained source and frozen-fit reservations that
were acquired before the observed call. Reattaching a live reservation is
idempotent and does not reserve its bytes twice. Reservations keep weak links to
observation recorders, so a long-lived prepared cache does not retain every past
receipt. Split and merged leases preserve the observed account balance.
