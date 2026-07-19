# ADR 0003: Categorical representation

- Status: Accepted
- Date: 2026-07-21

## Decision

Categoricals use dictionary-encoded `u32` category IDs with immutable domains.
Missingness is represented by validity bitmaps, not synthetic categories.
Contrasts are explicit model configuration and are stored in fitted artifacts.
Raw category IDs are never treated as numerical magnitudes.

## Consequences

- Contrast coding occurs during design compilation, not at column view time.
- Unknown levels fail by default unless an `Other` level is declared.
