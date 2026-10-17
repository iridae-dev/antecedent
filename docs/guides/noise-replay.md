# Replaying exogenous noise through a compiled model

A compiled structural model samples by drawing each node's exogenous noise inline.
Noise replay separates the two steps so a study can keep the noise, hold some
coordinates fixed, redraw others, and evaluate the model again, many times, without
cloning the model.

Rust surface: `antecedent_model::{NoiseTable, NoiseReplayer, NoiseSource, NoiseFix,
NoiseOrigin}`. Provenance record: `model.noise_replay`.

## The noise table

`NoiseTable::sample(view, n_rows, ctx)` draws one noise column per node. Each column is
drawn in blocks of `NOISE_ROW_BLOCK` rows from a stream keyed by `(draw, node, block)` in
a dedicated stream domain, so a node's noise does not depend on the other nodes, on the
overlay, or on the number of rows after its block. State-space and dynamic mechanisms,
whose noise is serially dependent across rows, are drawn as a single block. A soft
override draws from the override's own noise scale; a hard-set node whose mechanism is
unfitted needs no noise and is marked `Unused`.

Every column records where it came from: `Sampled { draw }`, `Fixed`,
`Abducted(NoiseInferenceMode)`, `Assumed` (prior-sampled for an absent factual column) or
`Unused`. `NoiseTable::from_abducted` wraps the noise of the counterfactual engine's
abduction; a categorical node reports `Posterior`, and that disclosure (and its
rank-preservation assumption) is carried by the table (`has_posterior_noise`). The
counterfactual crate maps the mode to `NoiseInferenceKind` with `From`.

## Fixing and randomizing

`fix` sets chosen coordinates to the node's reference noise, a constant, explicit
columns, or the same columns of another table. Categorical noise must lie strictly inside
`(0, 1)`; other noise must be finite. `randomize(nodes, draw)` redraws chosen
coordinates on a fresh stream family (`draw >= 1`; draw `0` is the sampled noise), leaving
every other column untouched.

## Replay

`NoiseReplayer::replay_into` evaluates every node in topological order from a table under
an overlay. The per-node precedence (hard set, stochastic policy, soft override or fitted
mechanism, additive shift) is the single function forward sampling uses, so the two cannot
diverge. Stochastic-policy nodes draw from a stream keyed by the caller's salt.
`NoiseReplayer::sample` is a table draw plus its replay, and replaying the table
reproduces the values bit for bit. The sequential-stream forward samplers
(`sample_observational`, `sample_with_overlay`) keep their published draws.

Cancellation is observed once per node. `replay_into` then returns the exact completed
prefix of nodes with `cancelled` set; `replay` and the table draws return
`ModelError::Cancelled`. The memory check (`n_rows x n_nodes x 8` bytes, doubled when a
table and its values are both kept) runs before allocating and refuses with
`ModelError::ResourceLimit`, reporting the required and permitted bytes.

## Coalitions

`NoiseSource::coalition(fixed, resampled, mask)` reads each node's column from `fixed`
when the mask is set and from `resampled` otherwise. Nothing is copied: replaying 512
coalitions costs 512 passes of the kernel on one replayer and one output buffer.
`NoiseReplayer::scratch_footprint` exposes the buffer growth so reuse can be asserted.
