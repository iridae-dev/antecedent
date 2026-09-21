#!/usr/bin/env bash
# rustfmt check for CI: exactly `cargo fmt --all -- --check`. Any non-zero
# exit fails, including a missing rustfmt component, an unparseable file or a
# bad rustfmt.toml, not only a formatting diff.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! cargo fmt --all -- --check; then
  echo "FAIL: cargo fmt --all -- --check did not pass" >&2
  exit 1
fi
echo "rustfmt gate OK"
