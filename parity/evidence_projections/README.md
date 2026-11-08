# Evidence source projections

Content-addressed projections, `<fingerprint>.json`, written when an evidence
manifest is published (`scripts/evidence_earning.py`) and read when it is checked
(`scripts/evidence_manifests.py`). A projection guards one source file: the length
of every retained segment, the line and column layout of every unexecuted body,
and one SHA-256 over the retained text. A manifest's `guarded_file` entry names
its projection by this fingerprint, so a projection shared by many records is
stored once.
