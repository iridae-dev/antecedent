# Finite arrived-study execution

**Suite path:** `conformance/proposals/arrival`

The independent binary SCM is X→Y with selection confined to root X.
The source experiment assigns X; Y=I(U<0.2+0.6X), U uniform on [0,1].
The shared Y mechanism identifies the target interventional distribution.
Counts (80,20) at X=0 and (20,80) at X=1 produce means 0.2 and 0.8 and
contrast 0.6. Rust uses 100 observations per arm; Python scales each arm to
250 observations. The observed count total must equal the original planned
sample size, rather than a caller-supplied row-count label.

The executing test consumes this pin, the original repair/ranking consumers,
checked transport/z-transport formula binders, and the original finite compiled
evaluator. Its artifact retains every original artifact and complete base/arrived
provider, counts, actual population/intervention/snapshot binding, full atom law,
proof, expression and measured work. Fresh Python consumption independently
reconstructs these inputs. Sampling uncertainty remains unmeasured; these tests
are acceptance examples, not calibration or interval validation.

## Expected summary

Top-level keys: `active_mean, active_probabilities, arm0_counts, arm1_counts, calibration, contrast, control_mean, inference, python_sample_size, rust_sample_size, schema_version, scm` (12 fields).
