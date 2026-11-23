# envelope_unidentified_mass

**Suite path:** `conformance/pag/envelope_unidentified_mass`

PAG generalized-adjustment envelope preserves unidentified completion mass.

The PAG is `x o-> y`. It has two completions, `x -> y` and `x <-> y`, each of weight
one. Neither carries an adjustment certificate, so the effect is **not identified by the
implemented adjustment method**: identified weight 0, unidentified weight 2. This is a
conservative refusal, not a claim that the effect is unidentifiable by every method and
not a claim of a zero effect: a directed edge in a MAG/PAG can represent influence
together with latent confounding. Two cases means two completions were examined, not that
either supplies an adjustment certificate.

Per-completion reasons (asserted by `crates/antecedent/tests/pag.rs::envelope_unidentified_mass`):

- `x -> y`: the edge is invisible in the MAG, so the MAG is not adjustment amenable.
- `x <-> y`: no generalized adjustment set satisfies the criterion on that completion.

## Expected summary

Top-level keys: `evidence_kind, evidence_proposition, evidence_roles, evidence_source, limitations, oracle, require_mass_accounted, tolerance_class` (8 fields).
