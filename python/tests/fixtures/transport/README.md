# Structural transport fixture

`not_certified_v1.cbor` is a Rust-generated, Python-framed schema-v1 certificate.
It retains an explicitly inconclusive result on a two-node graph and two source
selection diagrams. It contains no numerical or impossibility claim.

Regenerate intentionally with:

```sh
ANTECEDENT_WRITE_NOTCERT_FIXTURE="$PWD/python/tests/fixtures/transport/not_certified_v1.cbor" \
  cargo test -p antecedent-io --lib negative_meta_certificate_rejects_altered_scope_and_witness
```

Python independently consumes, re-exports and consumes it again in
`test_frozen_rust_not_certified_consumes_without_upgrade`.
