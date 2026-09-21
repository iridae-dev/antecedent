#!/usr/bin/env bash
# Publish-time verification, run first by both publish workflows.
#
# A tag push (or a workflow_dispatch) must not be enough to publish: the tag has
# to name a commit that is on main, whose committed tree carries the tag's
# version, and on which the `ci` workflow finished with every required job
# green. The RC gate is a local convention; this is what the runners enforce.
#
#   bash scripts/verify_release.sh tag VERSION      commit on main, tree at VERSION, CI green
#   bash scripts/verify_release.sh wheels DIR VERSION [RUN_ID]
#                                                   every wheel in DIR is antecedent-VERSION-*,
#                                                   and (when RUN_ID is given) built from this commit
#
# Needs `gh` with GH_TOKEN (actions: read), uv, and a full-history checkout.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

verify_tag() {
  local version="$1" sha run_id tmp
  sha="$(git rev-parse HEAD)"

  # workflow_dispatch may name any ref; publishing is tag-only.
  if [[ "${GITHUB_REF:-refs/tags/v${version}}" != "refs/tags/v${version}" ]]; then
    fail "ref ${GITHUB_REF} is not the tag v${version}; publish only from the release tag"
  fi

  git fetch --quiet origin main
  if ! git merge-base --is-ancestor "$sha" origin/main; then
    fail "commit $sha is not an ancestor of origin/main"
  fi

  # The tag decides what ships; the tree must already agree instead of being rewritten.
  bash scripts/set_version.sh --check "$version"

  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN
  run_id="$(gh run list --workflow ci.yml --commit "$sha" --status success \
    --json databaseId --jq '.[0].databaseId // empty')"
  if [[ -z "$run_id" ]]; then
    fail "no successful ci run on $sha (gh run list --workflow ci.yml --commit $sha)"
  fi
  gh run view "$run_id" --json headSha,jobs >"$tmp/run.json"
  local required
  required="$(uv run --quiet --project python --only-group dev python scripts/ci_workflow.py required-jobs)"
  # shellcheck disable=SC2086
  uv run --quiet --project python --only-group dev python scripts/ci_workflow.py \
    check-run "$tmp/run.json" --head-sha "$sha" $required
  echo "verified: $sha is on main, tree is at $version, ci run $run_id is green"
}

verify_wheels() {
  local dir="$1" version="$2" run_id="${3:-}" sha w seg n=0
  shopt -s nullglob
  for w in "$dir"/*.whl; do
    n=$((n + 1))
    # PEP 427 file names normalise the version ("2.0.0-rc.1" -> "2.0.0rc1"), so
    # compare with the separators removed.
    seg="$(basename "$w" | cut -d- -f2)"
    if [[ "$(basename "$w")" != antecedent-* || "${seg//[-._]/}" != "${version//[-._]/}" ]]; then
      fail "wheel $(basename "$w") does not carry version ${version}"
    fi
  done
  if [[ "$n" -eq 0 ]]; then
    fail "no wheels in $dir"
  fi
  if [[ -n "$run_id" && "$run_id" != "${GITHUB_RUN_ID:-}" ]]; then
    sha="$(git rev-parse HEAD)"
    if [[ "$(gh run view "$run_id" --json headSha --jq .headSha)" != "$sha" ]]; then
      fail "run $run_id did not build $sha; its wheels are not this release"
    fi
  fi
  echo "verified: $n wheel(s) carry version $version"
}

case "${1:-}" in
  tag)
    [[ $# -eq 2 ]] || fail "usage: $0 tag VERSION"
    verify_tag "$2"
    ;;
  wheels)
    [[ $# -ge 3 && $# -le 4 ]] || fail "usage: $0 wheels DIR VERSION [RUN_ID]"
    verify_wheels "$2" "$3" "${4:-}"
    ;;
  *)
    fail "usage: $0 tag VERSION | wheels DIR VERSION [RUN_ID]"
    ;;
esac
