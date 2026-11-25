#!/usr/bin/env bash
# Translate the production Rust kernels that the proofs in this package cover into
# Lean, with Charon and Aeneas at the versions pinned below.
#
#   AENEAS_DIR=/path/to/aeneas-release formal/rust/extract.sh
#
# AENEAS_DIR is an unpacked Aeneas release archive (aeneas, charon, charon-driver);
# its rust-toolchain names the nightly Charon compiles with. The translation of each
# kernel is written to AntecedentRust/Extracted/<Kernel>/ and committed; the proofs in
# AntecedentRust/<Kernel>.lean are checked against it. A Rust change to a covered item
# makes its assurance record stale until this is rerun and the proofs rebuilt.
set -euo pipefail

CHARON_VERSION="0.1.322"
AENEAS_RELEASE="nightly-2026.11.23-2e3d448"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
: "${AENEAS_DIR:?set AENEAS_DIR to an unpacked Aeneas ${AENEAS_RELEASE} release}"

actual="$("$AENEAS_DIR/charon" version | head -1 | cut -d' ' -f1)"
if [[ "$actual" != "$CHARON_VERSION" ]]; then
  echo "extract.sh: charon $actual, expected $CHARON_VERSION (Aeneas $AENEAS_RELEASE)" >&2
  exit 1
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
export CARGO_TARGET_DIR="${CHARON_TARGET_DIR:-$work/target}"

# EXTRACT_ONLY names one kernel to retranslate; the others are left as committed.
# extract <Kernel> <package> <charon args...>
extract() {
  local kernel="$1" package="$2"
  shift 2
  if [[ -n "${EXTRACT_ONLY:-}" && "$EXTRACT_ONLY" != "$kernel" ]]; then
    return 0
  fi
  local out="$here/AntecedentRust/Extracted/$kernel"
  (cd "$repo" && "$AENEAS_DIR/charon" cargo --preset=aeneas "$@" \
    --dest-file "$work/$kernel.llbc" -- -p "$package" >"$work/$kernel.charon.log" 2>&1) || {
    cat "$work/$kernel.charon.log" >&2
    exit 1
  }
  # FunsExternal.lean models the items translated as opaque (--opaque); it is written by
  # hand, so it survives retranslation.
  local external=""
  if [[ -f "$out/FunsExternal.lean" ]]; then
    external="$work/$kernel.FunsExternal.lean"
    cp "$out/FunsExternal.lean" "$external"
  fi
  rm -rf "$out"
  mkdir -p "$out"
  "$AENEAS_DIR/aeneas" -backend lean -split-files -subdir "AntecedentRust/Extracted/$kernel" \
    -namespace "AntecedentRust.Extracted.$kernel" -dest "$here" "$work/$kernel.llbc" \
    >"$work/$kernel.aeneas.log" 2>&1 || {
    cat "$work/$kernel.aeneas.log" >&2
    exit 1
  }
  if grep -q 'Generated the partial file' "$work/$kernel.aeneas.log"; then
    cat "$work/$kernel.aeneas.log" >&2
    echo "extract.sh: $kernel translated only partially" >&2
    exit 1
  fi
  # Source spans move with unrelated edits; the assurance record binds the items by
  # their tokens instead, so the committed translation drops them.
  sed -i.bak -E 's/, lines [0-9]+:[0-9]+-[0-9]+:[0-9]+//' "$out"/*.lean
  rm -f "$out"/*.bak "$out/FunsExternal_Template.lean"
  if [[ -n "$external" ]]; then
    cp "$external" "$out/FunsExternal.lean"
  fi
}

extract Standing antecedent-contract \
  --start-from 'antecedent_contract::claim::domain_change_allowed' \
  --start-from 'antecedent_contract::claim::identification_status_kept_or_lowered' \
  --include 'antecedent_core::identification' \
  --start-from 'antecedent_contract::claim::domain_upgraded' \
  --start-from 'antecedent_contract::claim::domain_departs' \
  --start-from 'antecedent_contract::claim::associational_derives_causal' \
  --start-from 'antecedent_contract::claim::non_value_derives_value' \
  --start-from 'antecedent_contract::claim::identification_slot_added' \
  --start-from 'antecedent_contract::claim::chain_retains'

# The obligation rule of TransformationReport::compose.
extract Transform antecedent-contract \
  --start-from 'antecedent_contract::transform::obligation_merge' \
  --start-from 'antecedent_contract::transform::obligation_kept_on_replace' \
  --start-from 'antecedent_contract::obligation::obligation_unresolved' \
  --include 'antecedent_core::assumption::AssumptionStatus'

# Revalidation classes: graph claims, provider trust and ledger results.
extract Revalidation antecedent-contract \
  --start-from 'antecedent_contract::composition_verification::revalidate::trust_move_is_numerical' \
  --start-from 'antecedent_contract::composition_verification::revalidate::revalidation_class' \
  --start-from 'antecedent_contract::composition_verification::graph::TrustStanding::of_provider' \
  --start-from 'antecedent_contract::evidence_ledger::ledger::loss_class' \
  --start-from 'antecedent_contract::evidence_ledger::ledger::status_step' \
  --include 'antecedent_core::recalc::Revalidation'

# The support severity order. Revalidation::worst and ChangeKind::revalidation are checked
# by Kani instead (formal/kani/src/revalidation.rs): translating them here would define
# Revalidation a second time beside the translation above.
extract Support antecedent-core \
  --start-from 'antecedent_core::response::SupportStatus::severity'

# The canonical framing of identities. The two standard-library sorts are opaque; their
# model is AntecedentRust/Extracted/Framing/FunsExternal.lean.
extract Framing antecedent-core \
  --start-from 'antecedent_core::framing::frame_word' \
  --start-from 'antecedent_core::framing::frame_count' \
  --start-from 'antecedent_core::framing::frame_bytes' \
  --start-from 'antecedent_core::framing::frame_presence' \
  --start-from 'antecedent_core::framing::frame_real_bits' \
  --start-from 'antecedent_core::framing::frame_words' \
  --start-from 'antecedent_core::framing::frame_u32s' \
  --start-from 'antecedent_core::framing::frame_digests' \
  --start-from 'antecedent_core::framing::dedup_sorted_digests' \
  --start-from 'antecedent_core::framing::frame_digest_set' \
  --start-from 'antecedent_core::framing::frame_digest_multiset' \
  --start-from 'antecedent_core::framing::frame_u32_multiset' \
  --opaque 'antecedent_core::framing::sort_digests' \
  --opaque 'antecedent_core::framing::sort_u32s'
