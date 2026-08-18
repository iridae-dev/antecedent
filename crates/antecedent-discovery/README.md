# antecedent-discovery

Static and temporal causal discovery. Static: PC, FCI/RFCI (→ PAG), GES
(→ CPDAG), DirectLiNGAM and NOTEARS (→ DAG), and Bayesian graph posteriors
(exact DAG posterior, structure/order MCMC, CI-screened and DBN posteriors).
Temporal: the PCMCI family (PCMCI, PCMCI+, LPCMCI, J-PCMCI+, regime-specific
RPCMCI) with compiled constraint masks (`required` / `forbidden` / `tiers`),
FDR control over the full MCI family, target-wise parallelism, and
`TemporalGraphReview` output.

Discovery returns equivalence classes and posteriors, not a single guessed
DAG; orientations are never invented downstream.
