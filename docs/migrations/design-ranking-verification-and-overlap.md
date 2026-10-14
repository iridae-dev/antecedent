# Design ranking: verification labels, overlap decisions and Monte Carlo streams

## What changed

1. A stored candidate whose signal was labelled `exact_request_verified` is consumed as `externally_attested`, with an explicit trust limit, unless the consumer supplies the verification receipt for that candidate (`consume_wire_with_receipts`). The artifact does not retain a receipt, so a stored label is a claim, not evidence. The label inside the signal identity is unchanged, so an honest verified artifact still replays and a resealed upgrade is still refused.
2. A body written with a newer layout version is refused as an unsupported version before it is decoded, in the design ranking reader and in the decision contract, result, robust and structural readers.
3. Whether a study's data overlap the prior's is read from the evidence ledger, not from a private intersection of id lists. A study derived from the prior's data under another id is refused as shared; a lineage the ledger cannot decide is refused as `evidence_overlap_unknown` (`evsi.source_overlap_unknown`, `design_ranking.source_overlap_unknown`, `prior_signal.observation_overlap_unknown`) and is never treated as disjoint. The legacy id-list entry point treats its lists as explicit declarations, so an empty list declares no reuse.
4. The Monte Carlo expected value of sample information draws each candidate's replicates from a stream keyed by the run seed, the candidate's semantic id and the replicate ordinal. A candidate's estimate no longer depends on which other candidates were evaluated, and candidates no longer share random numbers.

## Class

Migrated. Stored design rankings still decode and replay. Items 1 and 2 change what a consumer reports or refuses, not the layout. Item 4 changes the numbers of a freshly computed Monte Carlo estimate within the same tolerance; Monte Carlo results are not replayed from an artifact, so no stored value is refused.

## Who is affected

Consumers that read the trust label of a consumed design ranking, and callers that relied on a verified label surviving a round trip without a receipt. Callers that passed overlapping or unlabeled lineages and relied on them being treated as independent. Anyone comparing Monte Carlo expected-value estimates across releases at a fixed seed.

## How to regenerate

Nothing is required for stored artifacts. To retain a verified label, consume with the receipt for that candidate's provider object and request. To resolve an undecided overlap, declare the observations of the study and the prior sources in the evidence ledger and evaluate through the ledger-backed entry (`evaluate_evsi_with_evidence`).
