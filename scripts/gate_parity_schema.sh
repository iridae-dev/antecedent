#!/usr/bin/env bash
# Parity manifest schema gate: every [[capabilities]] row carries its required keys.
#
# Why this exists: the feature gates parse manifests with a regex `caps()` helper
# whose accessor takes an explicit default (`g("status", default=None)`). A row
# missing `status` therefore reads as None, matches none of the honesty checks,
# and passes every gate forever without ever being marked done/pending. Those
# parsers cannot detect an absent key by construction -- this gate is the
# schema-completeness check that closes that hole.
#
# It also pins the regex parser itself: the ids and statuses the gates' `caps()`
# recovers must agree with a real TOML parse, so a manifest whose layout drifts
# out from under the regex fails here instead of silently under-reporting.
#
# Run standalone, or via any feature gate / gate_release.sh, which all invoke it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
from pathlib import Path
import re
import sys
import tomllib

root = Path(".")

STATUS_VALUES = {"pending", "in_progress", "done"}
FACADE_VALUES = {"full", "thin"}
BASE_REQUIRED = ("id", "status")

# Inventory manifests and the extra keys each one requires beyond BASE_REQUIRED.
# Kept explicit rather than inferred from whichever keys the majority of rows
# happen to carry: the richer schema is what exposed the row that motivated this
# gate, so it has to be a stated contract, not a statistical accident.
MANIFESTS = {
    "parity/estimate.toml": ("group", "description", "owner"),
    "parity/discovery.toml": ("group", "description", "owner"),
    "parity/context.toml": ("group", "description", "owner"),
    "parity/bayesian.toml": (),
    "parity/pag.toml": (),
    "parity/gcm.toml": (),
    "parity/attribution.toml": (),
    "parity/design_state.toml": (),
    "parity/release.toml": (),
    "parity/response.toml": (),
}

# The parser every feature gate embeds. Reproduced verbatim so this gate checks
# what those gates actually see, not an idealized reading of the file.
def regex_caps(text: str):
    blocks = re.split(r"\n\[\[capabilities\]\]\n", text)[1:]
    out = []
    for b in blocks:
        def g(k, default=None):
            m = re.search(rf'^{k}\s*=\s*"([^"]*)"', b, re.M)
            if m:
                return m.group(1)
            m = re.search(rf'^{k}\s*=\s*(\d+)', b, re.M)
            return m.group(1) if m else default
        out.append({"id": g("id"), "status": g("status")})
    return out


problems = []

# Any manifest carrying capability rows must be enrolled here; a new inventory
# cannot join the repo and skip the schema contract.
for path in sorted(root.glob("parity/*.toml")):
    rel = path.as_posix()
    if rel in MANIFESTS:
        continue
    if re.search(r"^\[\[capabilities\]\]", path.read_text(), re.M):
        problems.append(f"{rel}: has capability rows but is not enrolled in gate_parity_schema.sh")

for rel, extra_required in MANIFESTS.items():
    path = root / rel
    if not path.exists():
        problems.append(f"{rel}: manifest missing")
        continue
    text = path.read_text()

    try:
        rows = tomllib.loads(text).get("capabilities", [])
    except tomllib.TOMLDecodeError as exc:
        problems.append(f"{rel}: not valid TOML: {exc}")
        continue

    if not rows:
        problems.append(f"{rel}: no [[capabilities]] rows")
        continue

    required = BASE_REQUIRED + tuple(extra_required)
    seen_ids = set()

    for i, row in enumerate(rows, 1):
        cid = row.get("id")
        label = cid if isinstance(cid, str) and cid.strip() else f"row #{i}"

        for key in required:
            if key not in row:
                problems.append(f"{rel}: {label} missing required key `{key}`")
            elif not isinstance(row[key], str) or not row[key].strip():
                problems.append(f"{rel}: {label} has empty/non-string `{key}`")

        status = row.get("status")
        if isinstance(status, str) and status.strip() and status not in STATUS_VALUES:
            problems.append(
                f"{rel}: {label} status={status!r} not in {sorted(STATUS_VALUES)}"
            )

        facade = row.get("python_facade")
        if facade is not None and facade not in FACADE_VALUES:
            problems.append(
                f"{rel}: {label} python_facade={facade!r} not in {sorted(FACADE_VALUES)}"
            )

        if isinstance(cid, str) and cid.strip():
            if cid in seen_ids:
                problems.append(f"{rel}: duplicate id `{cid}`")
            seen_ids.add(cid)

    # The gates' regex parser must recover the same rows the TOML parser sees.
    header_count = len(re.findall(r"^\[\[capabilities\]\]", text, re.M))
    scanned = regex_caps(text)
    if not (len(rows) == header_count == len(scanned)):
        problems.append(
            f"{rel}: row-count disagreement -- toml={len(rows)} "
            f"headers={header_count} gate-regex={len(scanned)}"
        )
    else:
        for row, seen in zip(rows, scanned):
            for key in ("id", "status"):
                if row.get(key) != seen[key]:
                    problems.append(
                        f"{rel}: gate regex reads {key}={seen[key]!r} for "
                        f"{row.get('id')!r} but TOML has {row.get(key)!r}"
                    )

if problems:
    print("parity manifest schema violations:")
    for p in problems:
        print(" -", p)
    sys.exit(1)

total = sum(
    len(tomllib.loads((root / rel).read_text())["capabilities"]) for rel in MANIFESTS
)
print(f"parity manifest schema: ok ({total} rows across {len(MANIFESTS)} manifests)")
PY
