#!/usr/bin/env bash
# Install antecedent for Read the Docs pdoc. Prefer a published wheel so RTD
# does not rustc the crate (~9 min historically). Compile from this checkout only
# when that exact version is not on PyPI yet.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="$(python -c "import tomllib; print(tomllib.load(open('${ROOT}/python/pyproject.toml', 'rb'))['project']['version'])")"

if python -m pip install "antecedent==${VERSION}"; then
  INSTALLED_VERSION="$(python -c 'import antecedent; print(antecedent.__version__)')"
  test "${INSTALLED_VERSION}" = "${VERSION}"
  echo "RTD: installed antecedent==${VERSION} from PyPI"
  exit 0
fi

echo "RTD: antecedent==${VERSION} not on PyPI; compiling ./python"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.85
PATH="${HOME}/.cargo/bin:${PATH}" python -m pip install "${ROOT}/python"
INSTALLED_VERSION="$(python -c 'import antecedent; print(antecedent.__version__)')"
test "${INSTALLED_VERSION}" = "${VERSION}"
