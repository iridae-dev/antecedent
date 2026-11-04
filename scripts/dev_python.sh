#!/usr/bin/env bash
# Build the Python extension as a DEVELOPMENT build and install it into the active
# environment.
#
# A development build runs routes a release build refuses as `not_supported` and lists what
# it ran under `result.would_refuse`; `antecedent.build_flavor()` returns "development".
# Never ship or publish what this builds, never write fixtures from it (artifacts it seals
# carry `build: development` and a release build refuses them).
#
# Usage: scripts/dev_python.sh [extra maturin args...]
#   Release-profile build (slower to compile, ~50x faster to run): scripts/dev_python.sh --release
set -euo pipefail
cd "$(dirname "$0")/../python"
maturin develop --locked --features development "$@"
python - <<'PY'
import antecedent

flavor = antecedent.build_flavor()
assert flavor == "development", f"expected a development build, got {flavor!r}"
print(f"antecedent build flavor: {flavor}")
PY
