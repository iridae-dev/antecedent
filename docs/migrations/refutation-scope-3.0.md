# Refutation scope metadata in 3.0

`antecedent_io::refutation_from_wire` returns `Result<RefutationReport, IoError>`: handle malformed or inconsistent declared target metadata rather than accepting it silently. Its decoded scope is descriptive only; decoding never recreates the private runtime authority of the operation that produced a report. A legacy report with no scope remains readable and retains its original serialized representation.

Whole-target and individual class-completion scopes name different targets. A completion report alone is not a verdict on an outer class or identified-set result. Existing reports without scope are historical unscoped evidence; they are not retroactively upgraded to checked scoped claims.

For Python, `ValidationView.passed` is now `bool | None`: `None` means checks ran but no whole-result verdict was established. Use `is True` or `is False` where the distinction matters. Scoped report views expose their descriptive scope and target identity; metadata is not proof of execution.
