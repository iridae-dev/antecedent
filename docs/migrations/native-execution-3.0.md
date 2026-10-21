# Shared execution accounts and native observations in 3.0

Batch workers now use the shared indexed scheduler and bounded inner contexts.
Numerical scratch uses the enclosing workflow admission account, including worker
execution. Standalone numerical components reserve their scratch bytes separately.
Admission peaks describe reservations, not process resident memory.

`Study::run_observed` and `PreparedStudy::estimate_retained_observed` retain work
reports and execution facts on success, failure and cancellation. Static results
expose observations under `performance.native_execution`, also available in Python.
Enclosing model fits are counted separately from their internal numerical solves.
Optional `observed_*` fields are absent where the corresponding boundary was not
measured. Instrumentation coverage remains incomplete across the full library.

Composite result and performance wires accept an optional `native_execution_v1`
record. Earlier artifacts without the field remain readable; unknown observation
formats are refused. Imported observations remain historical records. They grant
no execution or scientific authority. Licensing and calibration are unchanged.

This is the resource and observation foundation for the compiler integration
repair. It does not close H03/H04 or establish complete workflow resource accounting.
