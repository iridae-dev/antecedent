#!/usr/bin/env bash
# Public-API regression gate: every published crate's public surface (items,
# signatures, trait impls) is compared with the committed snapshot under
# parity/public_api/, built from rustdoc JSON on the pinned toolchain.
#
# An API change must be deliberate: regenerate with
#   bash scripts/gate_public_api.sh --write
# and review the snapshot diff. Run standalone or via scripts/gate_release.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 "$ROOT/scripts/gate_public_api.py" "$@"

# Every public facade entry names its Python entry or its reason; a documentation sentence that
# says an entry has no Python entry is refused while the registry binds one.
python3 "$ROOT/scripts/check_python_surface.py"
