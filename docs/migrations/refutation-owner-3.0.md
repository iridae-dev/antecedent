# Checked refutation ownership in 3.0

`StudyResult` exposes its scientific fields through `StudyResultView`. Field reads such as `result.identification` and `result.refutations` remain available; use `as_ref()` or clone a field when a previous call moved an owned value out of the result. Requesting mutable access to the view, or calling `estimate_mut`, permanently invalidates that result's checked refutation authority. A new checked execution produces a fresh owner. Descriptive results remain inspectable.

`certify_result_refutations` accepts the `TargetRefutationOutcome` returned by the checked prepared refit operation rather than a caller-constructed report slice. The outcome is tied to the exact result owner and execution data; mutating its exposed view invalidates its checked authority. Generic `TargetRefit` reports and serialized scope descriptors remain descriptive and do not acquire this private in-process authority.
