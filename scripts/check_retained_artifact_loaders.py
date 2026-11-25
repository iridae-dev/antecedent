#!/usr/bin/env python3
"""Every artifact loader offers a form that admits only the artifact its consumer retained.

Static: reads the Rust sources of the crates that load artifacts. A loader is a public function
or associated function whose first argument after any receiver is the artifact bytes (`&[u8]`)
or a path, that returns a `Result`, and that is neither `#[doc(hidden)]`, `#[deprecated]` nor
test-only. Each loader needs, in the same file, a bound form named `<loader>_expecting` (or
`<loader>_retained` when the loader's own name already ends in `_expecting` / `_expected`) that
takes a `RetainedArtifactIdentity` and whose body binds the bytes before it reads them: it is
`load_expecting(...)`, or it calls `bind_retained_artifact(...)` before it calls the loader, or it
delegates to another bound form. `load_expecting` and `bind_retained_artifact` go through
`antecedent_io_core::bound_identity::bind`, the comparison the `acceptance.retained_artifact`
assurance record proves.

The entries in EXEMPT are not artifact loaders; each names why.

Usage:
  check_retained_artifact_loaders.py              check the repository
  check_retained_artifact_loaders.py --list       print every loader and its bound form
  check_retained_artifact_loaders.py --self-test  the check rejects deliberately broken inputs
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CRATES = [
    "crates/antecedent-io-core/src",
    "crates/antecedent-io/src",
    "crates/antecedent-design/src",
]
FACADE_FILES = ["crates/antecedent/src/analysis/surface/io.rs"]

# Public functions that take artifact bytes but do not admit an artifact, by name.
EXEMPT = {
    "inspect": "reads the header for display; admits nothing",
    "inspect_within": "reads the header for display; admits nothing",
    "shape": "names the artifact shape; admits nothing",
    "shape_within": "names the artifact shape; admits nothing",
    "recognizes": "kind sniffing; admits nothing",
    "recognised_artifact_kind": "kind sniffing; admits nothing",
    "retain_disk_backed": "takes a spill directory, not an artifact",
    # compiled_study::open is a loader. open_expecting binds the invariant.
    "from_cbor": "generic CBOR codec under every loader",
    "decode_with_version": "version gate of the shared codec under every loader",
    "path_specific_from_wire": "wire conversion, not bytes",
    "path_specific_to_wire": "wire conversion, not bytes",
    "zstd_compress_chunked": "stream compressor",
    "cyclic_request_identity": "digests a request; admits nothing",
    "cyclic_decode_admission": "sizes the admission lease before decoding; admits nothing",
    "policy_program_callback_records": "lists callback records for the caller to supply",
    "inspect_array_fit_header": "header probe; admits nothing",
    "inspect_external_cyclic_artifact": "header probe; admits nothing",
    "inspect_provider_envelope": "header probe; admits nothing",
    "inspect_provider_envelope_within": "header probe; admits nothing",
    "read_metadata": "stream metadata probe; admits nothing",
    "accept_claim": "deprecated; accept_claim_expecting binds the retained claim id",
    "accept_claim_expecting": "binds the retained claim id (acceptance.retained_expected_identity)",
    "replicate_source": "calibration resampling of a source already admitted by its caller",
    "calibration_candidate": "calibration replay of a source already admitted by its caller",
    "describe_artifact": "describes a node for display; admits nothing",
    "detect_node_kind": "kind sniffing; admits nothing",
    "mean_decision_bytes": "produces bytes from an admitted contract",
    "bind_retained_artifact": "the binding check every bound form calls",
    "read_from": "container codec under every artifact loader; the loaders above it bind",
    "read_from_with_limits": "container codec under every artifact loader; the loaders above it bind",
    "read_selective": "container codec under every artifact loader; the loaders above it bind",
    "read_selective_with_limits": "container codec under every artifact loader; the loaders above it bind",
    "read_and_migrate": "container migration codec under the loaders; the loaders above it bind",
    "read_streaming": "chunked container reader that never holds the artifact whole",
    "zstd_decompress_chunked": "stream decompressor",
    "read_to": "chunked stream reader that never holds the stream whole",
    "source_content_digest": "rebuilds a source-witness content digest; admits nothing",
    "accept_source_body": "checks a catalog content digest the study opener already required",
    "decode_identification_product": "checks canonical identification-product bytes; the digest is identification_product_digest_wire, not an artifact admission",
}
# Calibration replays of a measured source, called on bytes the measured consumer admitted.
EXEMPT_PREFIX = {
    ("measured_inference/graph_candidate.rs", "replay_"),
    # The rollout producer binds the source and ranking bytes it is given to its declared
    # expectation; it builds an artifact, it does not admit one.
    ("rollout_artifact.rs", "new"),
    ("rollout_artifact.rs", "with_source_evidence"),
    # Opens a row spill file the process wrote itself, not an artifact.
    ("weighted_gaussian_capacity/rows.rs", "open"),
    # Validates a caller-owned stream under StreamBounds. Retained-identity loaders sit above it.
    ("stream_artifact.rs", "open"),
}
NOT_LOADER_NAME = re.compile(
    r"^(to_|write|encode|digest|hash|canonical|check_|validate_|is_|peek|container_kind|f64s|"
    r"update|measure|bytes$|payload_digest)"
)
FN_HEAD = re.compile(r"((?:#\[[^\]]*\]\s*|///[^\n]*\n\s*)*)pub (?:const |unsafe )?fn (\w+)\s*(<[^{(]*?>)?\s*\(", re.S)


class _Fn:
    """One `pub fn`: attributes, name, argument text, return text, end of the signature."""

    def __init__(self, src: str, m: re.Match) -> None:
        depth = 0
        i = m.end() - 1
        for i in range(m.end() - 1, len(src)):
            if src[i] == "(":
                depth += 1
            elif src[i] == ")":
                depth -= 1
                if depth == 0:
                    break
        tail = re.match(r"\s*(->\s*[^{;]+)?", src[i + 1 :])
        self.attrs, self.name, self.generics = m.group(1), m.group(2), m.group(3)
        self.args = src[m.end() : i]
        self.ret = tail.group(1) if tail else None
        self._end = i + 1 + (tail.end() if tail else 0)

    def groups(self) -> tuple:
        return self.attrs, self.name, self.generics, self.args, self.ret

    def end(self) -> int:
        return self._end


class FN:  # noqa: N801 - mirrors the regex interface it replaces
    @staticmethod
    def finditer(src: str):
        for m in FN_HEAD.finditer(src):
            yield _Fn(src, m)
BYTES_OR_PATH = re.compile(
    r":\s*&\[u8\]|:\s*Arc<\[u8\]>|:\s*Vec<u8>|:\s*Arc<Mmap>|&(?:std::path::)?Path\b|"
    r"AsRef<(?:std::path::)?Path>|:\s*(?:&mut )?R\b|impl (?:std::io::)?Read\b"
)
BINDS = ("load_expecting(", "bind_retained_artifact(")


def _split_args(args: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in args.replace("->", "\u2192"):
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def bound_name(name: str) -> str:
    if name.endswith("_expecting") or name.endswith("_expected") or "_expected_" in name:
        return name + "_retained"
    return name + "_expecting"


def _body(src: str, start: int) -> str:
    open_at = src.find("{", start)
    depth = 0
    for i in range(open_at, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[open_at + 1 : i]
    return ""


def _strip_tests(src: str) -> str:
    """Drop `#[cfg(test)]` modules so test helpers are not loaders."""
    out = src
    for m in reversed(list(re.finditer(r"#\[cfg\(test\)\]\s*(?:#\[[^\]]*\]\s*)*mod \w+\s*\{", out))):
        body = _body(out, m.end() - 1)
        out = out[: m.start()] + out[m.end() + len(body) + 1 :]
    return out


def loaders(rel: str, src: str) -> list[str]:
    found = []
    for m in FN.finditer(src):
        attrs, name, _generics, args, ret = m.groups()
        if any(tag in attrs for tag in ("doc(hidden)", "deprecated", "cfg(test)")):
            continue
        if not ret or "Result" not in ret:
            continue
        params = _split_args(" ".join(args.split()))
        params = [p for p in params if not re.match(r"&?('\w+ )?(mut )?self\b", p)]
        if not params or not BYTES_OR_PATH.search(params[0]):
            continue
        if NOT_LOADER_NAME.match(name) or name in EXEMPT:
            continue
        if any(rel.endswith(f) and name.startswith(p) for f, p in EXEMPT_PREFIX):
            continue
        if name.endswith("_retained") or (
            name.endswith("_expecting") and "RetainedArtifactIdentity" in args
        ):
            continue
        found.append(name)
    return found


def bound_forms(src: str) -> dict[str, list[str]]:
    """Every bound form in `src`: name -> bodies (one per impl that defines it)."""
    forms: dict[str, list[str]] = {}
    for m in FN.finditer(src):
        _attrs, name, _generics, args, _ret = m.groups()
        if "RetainedArtifactIdentity" in args:
            forms.setdefault(name, []).append(_body(src, m.end()))
    return forms


def form_problem(loader: str, bodies: list[str], forms: dict[str, list[str]]) -> str | None:
    for body in bodies:
        if body.lstrip().startswith(("antecedent_io_core::load_expecting(", "crate::retained_artifact::load_expecting(", "load_expecting(")) or re.match(
            r"\s*(?:[\w:]+::)?load_expecting\(", body
        ):
            continue
        bind_at = min((body.find(b) for b in BINDS if b in body), default=-1)
        if bind_at >= 0:
            call = re.search(rf"\b{re.escape(loader)}\s*\(", body)
            if call and call.start() < bind_at:
                return "reads the bytes before binding them"
            continue
        if any(re.search(rf"\b{re.escape(other)}\s*\(", body) for other in forms if other != loader):
            continue
        # Delegates to a bound form elsewhere, handing it the retained identity.
        if re.search(r"\w+_(?:expecting|retained)\s*\([^;]*\bretained\b", body, re.S):
            continue
        return "does not bind the bytes to the retained identity"
    return None


def problems(root: Path = ROOT) -> tuple[list[str], list[tuple[str, str, str]]]:
    errors: list[str] = []
    listed: list[tuple[str, str, str]] = []
    files = [p for d in CRATES for p in sorted((root / d).rglob("*.rs"))]
    files += [root / f for f in FACADE_FILES]
    for path in files:
        rel = path.relative_to(root).as_posix()
        if "/tests/" in rel or rel.endswith("tests.rs") or "test_support" in rel:
            continue
        src = _strip_tests(path.read_text(encoding="utf-8"))
        forms = bound_forms(src)
        for loader in loaders(rel, src):
            want = bound_name(loader)
            if want not in forms:
                errors.append(f"{rel}: {loader} has no bound form {want}(…, retained: &RetainedArtifactIdentity)")
                continue
            problem = form_problem(loader, forms[want], forms)
            if problem:
                errors.append(f"{rel}: {want} {problem}")
            listed.append((rel, loader, want))
    return errors, listed


def self_test() -> int:
    good = (
        "impl A {\n    pub fn decode(bytes: &[u8]) -> Result<Self, IoError> { todo!() }\n"
        "    pub fn decode_expecting(bytes: &[u8], retained: &RetainedArtifactIdentity)"
        " -> Result<Self, IoError> {\n        load_expecting(bytes, retained, Self::decode)\n    }\n}\n"
    )
    missing = "pub fn consume(bytes: &[u8]) -> Result<X, IoError> { todo!() }\n"
    late = (
        "pub fn consume(bytes: &[u8]) -> Result<X, E> { todo!() }\n"
        "pub fn consume_expecting(bytes: &[u8], retained: &RetainedArtifactIdentity) -> Result<X, E> {\n"
        "    let out = consume(bytes);\n    bind_retained_artifact(bytes, retained).map_err(E::from)?;\n    out\n}\n"
    )
    unbound = (
        "pub fn consume(bytes: &[u8]) -> Result<X, E> { todo!() }\n"
        "pub fn consume_expecting(bytes: &[u8], retained: &RetainedArtifactIdentity) -> Result<X, E> {\n"
        "    consume(bytes)\n}\n"
    )
    hidden = "#[doc(hidden)]\npub fn decode_parts(bytes: &[u8]) -> Result<X, E> { todo!() }\n"
    nested = (
        "pub fn decode_with(bytes: &[u8], supply: &dyn Fn(&R) -> Option<C>) -> Result<X, E> { todo!() }\n"
        "pub fn decode_with_expecting(bytes: &[u8], supply: &dyn Fn(&R) -> Option<C>,"
        " retained: &RetainedArtifactIdentity) -> Result<X, E> {\n    decode_with(bytes, supply)\n}\n"
    )
    cases = [
        (good, True), (missing, False), (late, False), (unbound, False), (hidden, True),
        (nested, False),
    ]
    failed = 0
    for src, ok in cases:
        forms = bound_forms(src)
        errs = []
        for loader in loaders("x.rs", src):
            want = bound_name(loader)
            if want not in forms or form_problem(loader, forms[want], forms):
                errs.append(want)
        if (not errs) != ok:
            failed += 1
            print(f"self-test: expected {'pass' if ok else 'failure'} for:\n{src}", file=sys.stderr)
    if bound_name("consume_expecting") != "consume_expecting_retained":
        failed += 1
    print("self-test OK" if not failed else f"self-test: {failed} case(s) wrong")
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    errors, listed = problems()
    if "--list" in argv:
        for rel, loader, want in listed:
            print(f"{rel}: {loader} -> {want}")
    for error in errors:
        print(f"FAIL: {error}")
    if errors:
        return 1
    print(f"retained artifact loaders OK: {len(listed)} loaders offer a bound form")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
