# Temporal class observation

**Suite path:** `conformance/response/temporal_class_observation`

Licensed 1.3/1.6 observation pairs on `TemporalCpdag` / `TemporalPag`
`ResponseCurve` / `InterventionResponse`, including Sequence overlays. The
observation mechanism is adjusted per completion (IPCW for Frequentist,
observed-data Gaussian likelihood for Bayesian), and the class-mass contract
is applied afterwards. Complete-data bands are never reused.

Licensed pairs: `Selected × OutcomeIndependentGiven`, `RightCensored` /
`LeftCensored × IndependentGiven([])` and `IndependentGiven([T])`. Delayed
entry, interval censoring, and truncation stay refused on incomplete classes.

Frequentist requested replicates run the outer circular-block bootstrap that
refits the observation nuisance and every horizon or Sequence overlay per
completion. A single identified completion carries the class band; a
multi-completion identified set withholds the class band while each atom
retains its own outer-block band.

The incomplete class is a `TemporalCpdag` with an off-path undirected mark
(`z@1 — t@1`) on the two-lag `t -> y` skeleton; the `TemporalPag` form uses
the same skeleton.

Consumed by `crates/antecedent/tests/temporal_class_observation.rs` and
`temporal_class_observation_does_not_reuse_complete_band` in
`crates/antecedent/tests/temporal_class_bayesian_envelope.rs`, plus
`python/tests/test_temporal_class_bayesian_envelope.py`.
