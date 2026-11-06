# Replayable joint sensitivity interval

The endpoint bootstrap has a separate version-1 interval artifact. The embedded
version-3 artifact retains its existing assumption-range contract and continues
to refuse a sampling interval in that range format.

Rust `JointSensitivityIntervalArtifact::checked` consumes the checked baseline,
then runs the existing endpoint-percentile procedure for the declared box, seed,
replicate count and level. `consume` replays the range and the interval, checks
counts and every numerical vector, and returns `JointIntervalConsumption`.
Python `antecedent.transport.advanced.export_joint_sensitivity_interval` returns
portable bytes in a `JointSensitivityIntervalArtifact`; its `replay()` method
performs independent consumption. An executed baseline is required for export.

The wire retains requested/successful/failed counts, replicate baselines and
range endpoints, endpoint quantiles, level, seed/stream version, kernel policy,
deviation box and source provenance. Consumers enforce encoded-size, replicate,
search and baseline limits before the corresponding work. Cancellation and
insufficient admission refuse; an interrupted or failed producer call exports no interval. Replicate losses remain
explicit and follow the original procedure’s acceptance policy.

`recorded_calibration` is historical producer metadata. `calibration` on the
consumed result is derived from the replayed construction and current records.
An edited historical status cannot promote current coverage. A content digest
binds premises and numbers; it does not authenticate the supplied baseline laws.

The interpretation remains the existing conditional endpoint-bootstrap target:
coverage of a fixed true deviation within the declared box, subject to the
pointwise bootstrap's validity. It is not a confidence interval for the whole
sensitivity set. Historical coverage only concerns the named zero-box and
positive-box designs. Current-branch evidence and the independent rational
vertex-enumeration checks remain owed after freeze. No scientific or calibration
lane was run to implement this artifact. Broader MNAR/MSM sampling composition
remains deferred.
