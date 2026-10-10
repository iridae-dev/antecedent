# Checked static and design recalculation oracles

**Suite path:** `conformance/recalculation/checked_static_design`

Oracle kinds: `enumerated_finite_law` for finite ADMG/multi-source response; `closed_form` for design-specific linear models. Fixtures construct their raw rows/laws independently of the selective engine. Neither frozen Antecedent outputs nor agreement between full/selective runs establishes scientific truth.

Static front-door rows enumerate a structural response contrast of0.282. The selected Verma graph yields do-means0.3275 and0.5025, contrast0.175, from the explicitly enumerated latent SCM. Multi-source transport compares both intervention means to the target SCM's exact enumeration. Additional refresh/full-run checks establish selective invalidation, not an independent oracle.

Design fixtures independently derive IV effect2, sharp RD jump2.5 and HC1 variance0.04*160/156, and the linear front-door path product0.5*3=1.5 with HC0 stacked variance0.4505. Successful least-squares observers count actual work, including IV/Anderson–Rubin auxiliary fits. No coverage claim follows from these algebraic checks.

Run `cargo test -p antecedent --test recalc_static --test recalc_design --offline`. Python bindings and fresh-process consumption are exercised in the corresponding `python/tests/test_recalc_static.py` and `test_recalc_design.py`; run with the current native extension. Artifact consumers verify premises and scientific bodies, not only a self-consistent checksum.

## Expected summary

Top-level keys: `frontdoor_discrete_contrast, frontdoor_hc0_variance, frontdoor_linear_effect, iv_effect, oracle_kind, rd_hc1_variance, rd_jump, verma_contrast, verma_do_means` (9 fields).
