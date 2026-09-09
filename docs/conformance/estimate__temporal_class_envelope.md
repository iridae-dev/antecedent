# TemporalCpdag / TemporalPag pulse envelope

**Suite path:** `conformance/estimate/temporal_class_envelope`

1.4 class-preserving Pulse / single-step Sustained. The source graph keeps
its incomplete class. Completions are TemporalDag members; each is identified
with `temporal.backdoor.unfolded` and mass-weighted.

A fully oriented TemporalCpdag / TemporalPag is the TemporalDag coordinate.

TemporalPag pins cover only DAG endpoint refinements, not all latent-confounded
MAG completions. Global PAG equivalence is not audited; the runtime diagnostic
records this restriction and cannot assert class-wide point identification.

## Expected summary

Top-level keys: `case, columns, cpdag, law, n, pag, query, schema_version` (8 fields).
