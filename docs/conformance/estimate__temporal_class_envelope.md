# TemporalCpdag / TemporalPag pulse envelope

**Suite path:** `conformance/estimate/temporal_class_envelope`

1.4 class-preserving Pulse / single-step Sustained. The source graph keeps
its incomplete class, including a fully oriented supplied graph. Completions
are TemporalDag (CPDAG) or directed/bidirected MAG (PAG) members; each is
identified and mass-weighted.

Completing the graph yourself is still the TemporalDag coordinate.

TemporalPag pins include a latent MAG witness (`latent.json`) whose known
lagged effect is 2. The older three-variable PAG interpretation is withdrawn:
its causal edge is invisible. Finite-window equivalence is audited; a capped
audit cannot confer class-wide point identification. Selection-variable
tail-tail edges remain outside this adjustment family.

## Identified multi-completion TemporalPag (1.9)

`identified_pag.json` is the positive TemporalPag evidence the withdrawn
three-variable graph no longer provides. The graph
`v@-1 o-o t@-1 o-o z@-1 o-o m@-1`, `t@-1 -> y@0`, `m@-1 -> y@0` has seven
stationary MAG completions: four identify the lag-1 pulse by adjusting
`z@-1` (direct effect, 1.9618 on the frozen series), two by adjusting nothing
(total effect through `z -> m -> y`, 2.7205), and one — no arrowhead into `t`,
so no visible edge out of it — stays unidentified (mass 1 of 7, reported and
never mixed in). The series is closed-form (no RNG); `identified_pag_reference.py`
rebuilds it in numpy and computes the equal-weight mixture over identified
completions, `2.214723444204995`.

Consumer: `temporal_pag_identified_multi_completion_pulse_and_sustained` in
`crates/antecedent/tests/temporal_class_envelope_numeric_pins.rs` — explicit
and accepted TemporalPag, Pulse and single-step Sustained, fresh and prepared,
`none`/`cheap`/`full`. It pins the point against the reference, the envelope
masses, and, with `bootstrap_replicates = 32`, a finite positive shared
circular-block SE with `estimate.temporal_class.frequentist.shared_block` and
without `estimate.envelope.se_omits_between_atom_variance`.

## Expected summary

Top-level keys: `case, columns, cpdag, law, n, pag, query, schema_version` (8 fields).
