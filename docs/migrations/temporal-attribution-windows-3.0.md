# Temporal attribution: observed windows and run-aware resampling

Temporal attribution now fits and scores complete observed lag windows. For a
required history of `h` rows, an eligible anchor starts at source tick `h` and
must satisfy the input analysis mask. Its lagged inputs come from the original
observations, not copies of the first observation. A mask controls anchor
eligibility; it does not itself declare that an earlier observation is missing.
Missing cells still follow the declared missing-data policy.

An explicit anomaly or unit row refers to the original source coordinate. A
request for a prefix row with insufficient history receives a typed refusal.
The published row coordinates remain original coordinates. For a graph class or
posterior, the compared members use the same source-anchor boundary, chosen
from the longest retained member history. Fitting can still refuse when the
remaining observations are insufficient for the declared estimator. Current
mechanism selection and fitting do not support user observation weights, so a
weighted input is refused rather than silently fitted without those weights.

A rollout starts from the real observed source tail, including when the fitted
window is shorter than the history. A missing auxiliary history cell is marked
unavailable; an active policy that actually reads it receives a typed refusal.
Gathered refits retain their source positions and block boundaries, including
repeated source ticks. Windows crossing a join are excluded without redrawing
the original sample.

The statistical program identity now records `complete_observed_derived_run_blocks_v2`.
The v2 block law keeps the primary stream's draw count and splits a nominal
block at disconnected runs. Additional segments use deterministic substreams
keyed by the replicate, primary draw, block, run and segment positions. A
bootstrap segment wraps within its own run; a global wrap starts a new derived
segment, even in a single run. Mechanism-change permutations preserve the
parent Fisher–Yates draws and use separate keyed streams to interleave split
segments. Bayesian serial-inflation calculations use lag pairs within runs.
Effective v2 weights and numerical answers can therefore differ from v1.

Historical artifacts without this field retain their original identity and
remain readable as historical evidence. The absent field is not evidence of v2,
and it does not authorize replay under the new convention. Existing unresolved
checked-attribution verification remains unresolved; this migration does not
supply historical numerical replay or authenticate producer origin.
