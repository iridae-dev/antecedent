# Evidence objects

The content-addressed store behind scientific passes, calibration measurements and
probe attestations, written by `scripts/evidence_earning.py` and read by
`scripts/evidence_manifests.py` (`scripts/evidence_objects.py` owns the format).

Each `<aa>.txt` shard holds one object per line, `<sha256> <canonical JSON>`, for
every address that begins with `aa`. The address is the SHA-256 of the canonical
JSON (sorted keys, compact separators, UTF-8, no non-finite numbers). A value
longer than 512 bytes is stored as its own object and referenced as
`{"@": address}`, and a long list is stored as content-defined chunks,
`{"@chunks": [...]}`, so a value shared by many passes is stored once. Writers
append under a lock on the shard, and every line is checked against its address
when it is read.

`../passes.json` maps each `pass:<kind>:<key>` to the address of its stored
manifest. What a manifest covers is described in
[Earning and checking the records](../../docs/guides/scientific-verification.md#earning-and-checking-the-records).
