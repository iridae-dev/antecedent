# Evidence input groups

Content-addressed input groups, `<sha256>.json`, written when an evidence manifest
is published (`scripts/evidence_earning.py`) and expanded when it is checked
(`scripts/evidence_manifests.py`). A group holds the inputs of one directory, or
one kind of named input, sorted by kind and path. A stored manifest lists the
addresses of its groups, so the inputs many records share are stored once; the
address is the SHA-256 of the group's canonical JSON.
