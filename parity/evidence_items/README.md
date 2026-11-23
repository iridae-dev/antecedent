# Evidence item tables

Content-addressed item tables, `<sha256>.json`, written when an evidence manifest
is published (`scripts/evidence_earning.py`) and read when it is checked
(`scripts/evidence_items.py`). A table lists the item locators of one Rust source
file in order (`scripts/rust_items.py`). A manifest's `rust_items` entry names its
file's table and selects the items, and the parts of them, the record depends on
by index, so a table is stored once for every record and file version with the
same item structure.
