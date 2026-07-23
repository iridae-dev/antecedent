# prior_bank_conflict_shrink

**Suite path:** `conformance/bayesian/prior_bank_conflict_shrink`

ConflictPolicy shrinks α when prior-PPC / KL signals indicate conflict;
no-conflict signals leave α unchanged.

Conflict case: `p = 0.001 ≤ p_min` ⇒ α' = 0.
No-conflict case: `p = 0.4`, `kl = 0` ⇒ α' = α.

## Expected summary

Top-level keys: `alpha, conflict, kl_scale, no_conflict, notes, p_min` (6 fields).
