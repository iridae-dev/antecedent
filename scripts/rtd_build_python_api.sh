#!/usr/bin/env bash
# Build pdoc HTML into $READTHEDOCS_OUTPUT/html/python (Read the Docs post_build).
# Locally: READTHEDOCS_OUTPUT=./site bash scripts/rtd_build_python_api.sh
set -euo pipefail

OUT="${READTHEDOCS_OUTPUT:?READTHEDOCS_OUTPUT unset}/html/python"
mkdir -p "${OUT}"
WORK="$(mktemp -d)"
cleanup() { rm -rf "${WORK}"; }
trap cleanup EXIT

cd "${WORK}"
python -c 'import antecedent; print(antecedent.__file__, getattr(antecedent, "__version__", "?"))'
python -m pdoc antecedent -o "${OUT}"
ls -la "${OUT}" | head -n 30
test -f "${OUT}/antecedent.html"
echo "pdoc ok → ${OUT}/antecedent.html"
