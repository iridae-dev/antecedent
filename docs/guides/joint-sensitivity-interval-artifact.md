# Replay a sensitivity interval

The [joint mechanism sensitivity](joint-mechanism-sensitivity.md) range tells you how far a
transported effect could move if the outcome model and the shared parent distribution deviate
from the source evidence by up to a declared amount. When the source law is a sample of counts,
that range also carries sampling noise. This page shows how to put an interval on that noise,
seal it as bytes, and have someone else check it without trusting you.

## When to use it

Use it when you report a sensitivity range from counted data and a reader, a reviewer or a later
job needs to confirm the interval. The artifact holds the declared box, the seed, the replicate
count and the level. A second party replays all of it from the bytes and gets either the same
numbers or a refusal.

If your source law is an exact supplied law with no counts, there is no sampling uncertainty and
the interval refuses with `exact_supplied_law_no_sampling_uncertainty`.

## Walk-through

The example below builds a small counted transport problem, runs the baseline, declares a
deviation box, exports the interval, and replays it with nothing but the bytes.

<!-- example: examples/python/joint_sensitivity_interval.py -->

The steps are:

1. Prepare the stage and run `prepared.estimate()`. Export needs an executed baseline and
   refuses without one.
2. Declare the box with `JointDeviation`, for example `{"outcome_kernel": 0.2,
   "shared_parent_marginal": 0.3}`.
3. Call `export_joint_sensitivity_interval(prepared, box, seed=..., replicates=..., level=...)`.
   It runs the endpoint bootstrap once with that seed and returns a
   `JointSensitivityIntervalArtifact`. The same inputs and seed give the same bytes.
4. Call `artifact.replay()`, or `consume_joint_sensitivity_interval(bytes)`. Replay reruns the
   baseline, the exact range and the seeded bootstrap, and compares every number and count with
   the stored ones. It returns the result only when they agree. `ant.load(artifact)` reads the
   same bytes through the same check.

From Rust, `JointSensitivityIntervalArtifact::checked` builds the artifact from a checked
baseline and `consume` replays it into a `JointIntervalConsumption`.

## What the replayed result holds

- The declared box, the seed and the stream version, and the kernel policy used.
- The level, the endpoint interval and the zero-deviation percentile interval.
- The requested, successful and failed replicate counts, with each replicate's baseline and range
  endpoints. A replicate that failed stays visible as a failure; it is never replaced.
- Two calibration fields. `recorded_calibration` is what the producer saw when it exported, kept
  as history. `calibration` is derived during the replay from the replayed construction and the
  coverage records that exist now. Its `status` is one of `calibrated`, `outside_measured_scope`,
  `measured_below_nominal` or `not_measured`. Editing the recorded status cannot raise the
  current one, and fewer than 199 replicates or another level is computed and annotated, not
  refused.

## What it claims

The interval is a conservative endpoint percentile bootstrap. Its target is coverage of a fixed
true deviation inside the declared box, if the pointwise bootstrap is valid for the data.

It does not claim:

- a confidence interval for the whole sensitivity set;
- that the declared box contains the true deviation;
- that the baseline laws are right. The content digest binds the premises and numbers, and
  replay shows they follow from those premises. Neither authenticates the supplied laws, and a
  producer who changes the laws consistently produces bytes that replay.

Coverage records measure only the named zero-box and positive-box designs. Outside those, the
calibration status says so. Combining this interval with missing-not-at-random or marginal
sensitivity models is not covered.

## Refusals and limits

- A replicate count outside 2 to 2000 or a level outside (0, 1) refuses at export with
  `joint_sensitivity.bounds_exceeded`. A level of 0.95 with 199 or more replicates is the
  measured setting.
- Export before the baseline has run refuses.
- An interrupted or failed export returns no artifact. Cancellation and insufficient memory
  admission refuse on export and on replay.
- Edited bytes, a bad frame, an unsupported procedure or stream version, or a changed number
  refuse on replay. A truncated artifact fails to decode.
- Replay enforces its own ceilings before doing the work: encoded size, replicates (`max_replicates`,
  default 2000), search operations, search depth and memory. A stored request above a ceiling
  refuses with `route_not_supported` and `joint_sensitivity.consumer_limits`. Raise a ceiling
  only if you accept the cost of that replay.

The assumption range itself keeps its own [range artifact](joint-mechanism-sensitivity.md#artifact-version-3),
which carries no sampling interval and refuses an interval in that format. This interval
artifact is a separate wire frame, `joint_sensitivity_interval_v1`.
