# Fuzz targets

Build with nightly + cargo-fuzz:

```bash
cargo +nightly install cargo-fuzz
cargo +nightly fuzz run schema_names
```

Targets exercise core schema construction; they must not import DoWhy/Tigramite
source.
