# Evidence dependency manifests

Content-addressed manifests, `<sha256>.json`, written when scientific passes,
calibration measurements and probe attestations are earned
(`scripts/evidence_earning.py`) and checked by `scripts/evidence_manifests.py`.
The address is the SHA-256 of the canonical JSON (sorted keys, compact
separators, UTF-8, no non-finite numbers). What a manifest covers is described in
[Earning and checking the records](../../docs/guides/scientific-verification.md#earning-and-checking-the-records).
