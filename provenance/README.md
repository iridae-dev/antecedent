# Algorithm provenance

Every substantive algorithm must have a machine-readable provenance record in
this directory. See DESIGN.md §27 and ADR 0008.

Copy [`_template.toml`](_template.toml) to a descriptive name
(e.g. `discovery.pcmci.toml`) and fill all fields truthfully.

## Rules

- Disclose prior exposure to upstream implementations.
- Set `source_translation = false` and `copied_* = false` for clean-room work.
- Cite papers with DOI and relevant sections.
- List independent test sources (synthetic SCMs, paper examples, black-box
 comparison against pinned baselines).
- Never commit upstream GPL source, translated GPL tests, or fixtures with
 unclear redistribution status.
