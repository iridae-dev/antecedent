#!/usr/bin/env bash
# Docs vs support-matrix honesty: capabilities/comparison must point at the
# matrix and must not treat closed/n-a cells as licensed public analyze support.
#
# Run standalone or via scripts/gate_release.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# --self-test: each broken input, applied alone to an overlay of the repo,
# must fail this gate with the expected message.
if [[ "${1:-}" == "--self-test" ]]; then
  exec python3 "$ROOT/scripts/selftest_cases.py" docs
fi

python3 - <<'PY'
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

root = Path(".")
fail: list[str] = []

DOCS = ("docs/capabilities.md", "docs/comparison.md")
MATRIX_LINK = "support-matrix.md"

# Same-sentence / same-paragraph hedges: the claim is inventory or refusal,
# not a public analyze license.
HEDGE = re.compile(
    r"refus(?:e|es|ed)|not licensed|not a licensed|inventory|support matrix",
    re.I,
)

# Overclaims that read as licensed analyze support unless already hedged: three
# fixed phrases, plus one pattern per closed rule of the matrix
# (parity/support_closed.toml: `queries` x `graph_classes`, rules that narrow no
# other axis): a sentence naming a closed query and a closed graph class next to
# "support" is a support claim about a cell the matrix refuses.
OVERCLAIMS = [
    re.compile(r"PAG\s+responses?", re.I),
    re.compile(r"PAG\s+curve", re.I),
    re.compile(r"we\s+support\s+PAG", re.I),
]
CLASS_WORDS = {"Dag": r"DAGs?", "Admg": r"ADMGs?", "Cpdag": r"CPDAGs?", "Pag": r"PAGs?"}
SUPPORT_VERB = re.compile(r"\bsupport(?:s|ed|ing)?\b", re.I)


def query_words(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "[ _-]?", name)


closed_pairs: list[tuple[re.Pattern[str], str]] = []
for rule in tomllib.loads((root / "parity/support_closed.toml").read_text()).get("closed", []):
    if not rule.get("queries") or not rule.get("graph_classes"):
        continue
    if any(k not in ("queries", "graph_classes", "reason") for k in rule):
        continue
    for q in rule["queries"]:
        for g in rule["graph_classes"]:
            if g in CLASS_WORDS:
                closed_pairs.append(
                    (
                        re.compile(rf"(?=.*\b{query_words(q)}\b)(?=.*\b{CLASS_WORDS[g]}\b)", re.I | re.S),
                        f"{q} x {g}",
                    )
                )


def paragraphs(text: str) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    start = 0
    for m in re.finditer(r"\n[ \t]*\n", text):
        out.append((start, m.start(), text[start : m.start()]))
        start = m.end()
    out.append((start, len(text), text[start:]))
    return out


def sentences(para: str, para_start: int) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    start = 0
    for m in re.finditer(r"(?<=[.!?])\s+", para):
        out.append((para_start + start, para_start + m.start(), para[start : m.start()]))
        start = m.end()
    out.append((para_start + start, para_start + len(para), para[start:]))
    return out


def containing(
    spans: list[tuple[int, int, str]], pos: int
) -> tuple[int, int, str] | None:
    for span in spans:
        if span[0] <= pos < span[1]:
            return span
    return None


for rel in DOCS:
    path = root / rel
    if not path.is_file():
        fail.append(f"{rel}: missing")
        continue
    text = path.read_text()
    if MATRIX_LINK not in text:
        fail.append(f"{rel}: does not link to {MATRIX_LINK}")

    paras = paragraphs(text)
    for pat in OVERCLAIMS:
        for m in pat.finditer(text):
            para_span = containing(paras, m.start())
            para = para_span[2] if para_span else ""
            sent = para
            if para_span is not None:
                sent_span = containing(sentences(para, para_span[0]), m.start())
                if sent_span is not None:
                    sent = sent_span[2]
            if HEDGE.search(sent) or HEDGE.search(para):
                continue
            snippet = re.sub(r"\s+", " ", m.group(0))
            fail.append(
                f"{rel}: overclaim {snippet!r} without refuse/not licensed/"
                f"inventory/support matrix nearby"
            )

    # Closed cells: no unhedged sentence may pair a closed query with a closed
    # graph class and a support verb.
    for para_start, _, para in paras:
        for s_start, _, sent in sentences(para, para_start):
            if not SUPPORT_VERB.search(sent) or HEDGE.search(sent) or HEDGE.search(para):
                continue
            for pattern, cell in closed_pairs:
                if pattern.search(sent):
                    fail.append(
                        f"{rel}: sentence claims support for closed cell {cell} without "
                        f"refuse/not licensed/inventory/support matrix nearby: "
                        f"{re.sub(chr(10), ' ', sent.strip())[:100]!r}"
                    )
                    break

# Counterfactual is licensed on one cell; remaining coordinates stay refused.
caps = root / "docs/capabilities.md"
if caps.is_file():
    text = caps.read_text()
    if not re.search(r"Counterfactual", text):
        fail.append("docs/capabilities.md: Counterfactual is not mentioned")
    if re.search(r"analyze[`']? refuses `?Counterfactual", text):
        fail.append(
            "docs/capabilities.md: Counterfactual is licensed; "
            "do not claim analyze refuses the query type"
        )
    licensed = re.search(
        r"Counterfactual[\s\S]{0,400}(licensed|explicit)[\s\S]{0,200}(Frequentist|none)",
        text,
        re.I,
    )
    remaining = re.search(
        r"Counterfactual[\s\S]{0,800}(accepted|nested|Bayesian|cheap/full)[\s\S]{0,200}refus",
        text,
        re.I,
    )
    if not licensed:
        fail.append(
            "docs/capabilities.md: licensed Counterfactual cell "
            "(explicit Frequentist / validation none) is not described"
        )
    if not remaining:
        fail.append(
            "docs/capabilities.md: remaining Counterfactual refusals "
            "(accepted/nested/Bayesian/cheap/full) are not named"
        )

# The current release notes are part of the 0.9 public contract, not an
# optional changelog paraphrase.  Verify that their refusal section accounts
# for every root query with zero licensed cells; otherwise a newly added root
# query (or a removed license) can leave the release notes silently stale.
cargo = tomllib.load(open(root / "Cargo.toml", "rb"))
version = cargo["workspace"]["package"]["version"]
preparation = root / "docs" / "release-notes" / "preparation.toml"
if preparation.is_file():
    target = tomllib.loads(preparation.read_text()).get("target_version")
    if not isinstance(target, str) or not re.fullmatch(r"\d+\.\d+\.\d+", target):
        fail.append("docs/release-notes/preparation.toml has no valid target_version")
        target = version
else:
    target = version
notes_path = root / "docs" / "release-notes" / f"v{target}.md"
if not notes_path.is_file():
    fail.append(f"{notes_path}: current release notes missing")
else:
    notes = notes_path.read_text()
    if "../support-matrix.md" not in notes:
        fail.append(f"{notes_path}: does not link the authoritative support matrix")
    refusal_heading = "## Explicit refusals"
    if refusal_heading not in notes:
        fail.append(f"{notes_path}: missing {refusal_heading!r}")
    else:
        refusal_text = notes.split(refusal_heading, 1)[1].split("\n## ", 1)[0]
        axes = tomllib.load(open(root / "parity/support_axes.toml", "rb"))
        licensed = tomllib.load(open(root / "parity/support_licensed.toml", "rb")).get(
            "cell", []
        )
        licensed_queries = {row["query"] for row in licensed}
        zero_cell_queries = set(axes["queries"]) - licensed_queries
        for query in sorted(zero_cell_queries):
            if f"`{query}`" not in refusal_text:
                fail.append(
                    f"{notes_path}: zero-cell root query {query!r} is absent from "
                    "the explicit-refusals section"
                )

# docs/short-series-thresholds.md is a measured table the licensed-cell prose quotes
# ("below 155 score effective rows"). The Rust constants are the source: the table's
# rows and every threshold quoted next to a link to the page must equal them.
threshold_doc = root / "docs/short-series-thresholds.md"
# Arms like `Self::Mediation | Self::Sequential => 40.0` name several families.
rust_thresholds = {
    name: int(m.group(2))
    for m in re.finditer(
        r"((?:Self::\w+\s*\|\s*)*Self::\w+)\s*=>\s*(\d+)\.0",
        (root / "crates/antecedent-estimate/src/temporal_block.rs").read_text(),
    )
    for name in re.findall(r"Self::(\w+)", m.group(1))
}
FAMILY_ROWS = {
    "SingleWindow": "single-window adjustment",
    "Mediation": "temporal mediation",
    "Sequential": "multi-step sequential",
    "Mixture": "multi-atom mixture",
}
if set(rust_thresholds) != set(FAMILY_ROWS):
    fail.append(
        "crates/antecedent-estimate/src/temporal_block.rs: min_effective_rows arms "
        f"{sorted(rust_thresholds)} differ from the families this gate pins {sorted(FAMILY_ROWS)}"
    )
else:
    table = {
        m.group(1).strip(): int(m.group(2))
        for m in re.finditer(r"^\|\s*([^|]+?)\s*\|[^|]*\|\s*(\d+)\s*\|\s*$", threshold_doc.read_text(), re.M)
    }
    for family, row in FAMILY_ROWS.items():
        if table.get(row) != rust_thresholds[family]:
            fail.append(
                f"docs/short-series-thresholds.md: {row!r} threshold is {table.get(row)}, "
                f"but {family} is {rust_thresholds[family]} in temporal_block.rs"
            )
    quoted = set()
    for cell in tomllib.loads((root / "parity/support_licensed.toml").read_text()).get("cell", []):
        limits = str(cell.get("limitations", ""))
        quoted |= {
            int(n)
            for n in re.findall(r"(?:below|fall below)\s+(\d+)(?:\s+score)?(?:\s+effective\s+rows)?[^.]{0,120}short-series-thresholds", limits)
        }
    for n in sorted(quoted - set(rust_thresholds.values())):
        fail.append(
            f"support_licensed.toml quotes a short-series threshold of {n}, "
            f"but temporal_block.rs has {sorted(set(rust_thresholds.values()))}"
        )

claims = tomllib.loads((root / "parity/claims.toml").read_text())
products = tomllib.loads((root / "parity/python_products.toml").read_text())
for claim in claims.get("claim", []):
    licensed = root / claim["licensed_by"]
    if not licensed.is_file():
        fail.append(f"claims.toml {claim['id']}: licensed_by {claim['licensed_by']} missing")
    if claim["id"] == "reusable_headline":
        for row in products.get("route", []):
            if not row.get("retains") and not row.get("reason"):
                fail.append(
                    f"python_products.toml route {row.get('kind')}/{row.get('data')}/"
                    f"{row.get('structure')}: retains=false without reason"
                )
    sentence = claim["sentence"]
    for rel in claim.get("files", []):
        text = (root / rel).read_text()
        count = text.count(sentence)
        if count != 1:
            fail.append(f"{rel}: claim {claim['id']} occurs {count} times (want 1)")

# A user-doc sentence that asserts statistical strength (calibrated, doubly robust,
# unbiased, exact DAG posterior, complete algorithm) must either hedge itself or be
# pinned in claims.toml with the registry file that carries its evidence.
STRENGTH = re.compile(
    r"(?i)\b(calibrated|doubly robust|unbiased|exact DAG posterior|complete algorithm)\b"
)
HEDGE = re.compile(
    r"(?i)\b(not|no|never|cannot|incomplete|refus\w*|without|unless|only|declared|"
    r"unlicensed|neither|nor|prove[sd]?)\b|n't"
)
for rel in ("README.md", "docs/capabilities.md", "docs/python-workflow.md"):
    pinned = [
        " ".join(c["sentence"].split())
        for c in claims.get("claim", [])
        if rel in c.get("files", [])
    ]
    for para in re.split(r"\n\s*\n", (root / rel).read_text()):
        flat = " ".join(para.split())
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z`*(|])", flat):
            if (
                STRENGTH.search(sentence)
                and not HEDGE.search(sentence)
                and not any(p in sentence for p in pinned)
            ):
                fail.append(
                    f"{rel}: unhedged strength claim not pinned in parity/claims.toml: "
                    f"{sentence[:120]!r}"
                )


def changelog_current(text: str) -> str:
    # The current version's section only; earlier sections are frozen history.
    heads = [f"## [{target}]", f"## {target}"]
    for head in heads:
        if head in text:
            rest = text.split(head, 1)[1]
            cut = re.search(r"\n## [\[0-9]", rest)
            return rest if cut is None else rest[: cut.start()]
    fail.append(f"CHANGELOG.md: no {heads[0]} or {heads[1]!r} section to scan")
    return ""


for row in claims.get("forbidden", []):
    pat = re.compile(row["pattern"])
    for rel in row.get("files", []):
        path = root / rel
        if not path.is_file():
            fail.append(f"{rel}: listed in claims.toml forbidden files but missing")
            continue
        text = path.read_text()
        if rel == "CHANGELOG.md" and row.get("changelog_section") == "current":
            text = changelog_current(text)
        for m in pat.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            fail.append(
                f"{rel}: forbidden {row.get('id', 'pattern')} {m.group(0)!r} "
                f"(line {line} of the scanned text)"
            )

# Prose that quotes how many licensed cells lack a coverage measurement must
# quote the registry's counts.
licensed_cells = tomllib.load(open(root / "parity/support_licensed.toml", "rb")).get("cell", [])
not_measured = sum(
    1 for c in licensed_cells if c.get("calibration_reason") == "estimator_grid_not_measured"
)
no_interval = sum(
    1 for c in licensed_cells if c.get("calibration_reason") == "no_interval_reported"
)
for rel, phrases in (
    (
        "docs/guarantees.md",
        (
            f"Of the {len(licensed_cells)} licensed cells",
            f"{not_measured} have no coverage measurement",
            f"{no_interval} report no interval",
        ),
    ),
    (
        "README.md",
        (
            f"of the {len(licensed_cells)} licensed cells, {not_measured} have no",
            f"{no_interval} report no interval",
        ),
    ),
):
    text = (root / rel).read_text()
    for phrase in phrases:
        if phrase not in text:
            fail.append(f"{rel}: calibration-coverage count must read {phrase!r}")

# The temporal-response short-series threshold is quoted in prose; the prose
# must carry the constant the code warns on.
dispersion = root / "crates/antecedent-estimate/src/temporal_response_dispersion.rs"
threshold = re.search(
    r"pub const RESPONSE_SHORT_SERIES_ROWS: f64 = ([0-9.]+);", dispersion.read_text()
)
if threshold is None:
    fail.append(f"{dispersion}: RESPONSE_SHORT_SERIES_ROWS not found")
else:
    rows = threshold.group(1).removesuffix(".0")
    for rel, phrase in (
        ("docs/causal-responses.md", f"fewer than {rows} effective rows"),
        ("docs/capabilities.md", f"short_series` warns below {rows}"),
    ):
        if phrase not in (root / rel).read_text():
            fail.append(
                f"{rel}: temporal-response short-series threshold must read "
                f"{phrase!r} (RESPONSE_SHORT_SERIES_ROWS = {threshold.group(1)})"
            )

if fail:
    print("Docs support-matrix gate FAILED:")
    for item in fail:
        print(f" - {item}")
    sys.exit(1)

print(
    "Docs support-matrix OK (capabilities/comparison link the matrix; current "
    "release notes enumerate every zero-cell root query; Counterfactual "
    "license + remaining refusals are described; no unhedged PAG overclaim and "
    "no unhedged support claim on a closed query x graph-class cell; "
    "short-series thresholds match temporal_block.rs; claims.toml sentences and "
    "forbidden patterns hold)"
)
PY

python3 scripts/short_series_thresholds.py check
