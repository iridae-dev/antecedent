#!/usr/bin/env bash
# Lint-allow policy for the three lints that can hide a real defect.
#
# `clippy::cast_possible_truncation`, `clippy::cast_sign_loss` and `clippy::float_cmp`
# flag a silent wrap or clamp and an exact float comparison. A blanket `#![allow]` of
# any of them in library code lets every later cast in the file through unreviewed, so:
#
#   * library code (anything outside tests/, benches/, examples/) may not carry an
#     unconditional inner `#![allow(...)]` of them; an allow sits on the smallest item
#     or statement that needs it and states why in `reason = "..."`;
#   * unit tests may allow them file-wide only as `#![cfg_attr(test, allow(..., reason = ..))]`;
#   * integration-test, bench and example targets may allow them file-wide, with a reason.
#
# Every allow of these lints, at any scope, must carry a `reason` (an item-level allow may
# instead state it in a comment on the same line). `#[expect(...)]` suppresses the lint
# exactly as `#[allow(...)]` does and is held to the same rules.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 "$ROOT/scripts/gate_lint_allows.py"
