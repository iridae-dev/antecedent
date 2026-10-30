"""Source analysis for scripts/check_promotion_records.py.

Pure functions over Rust and Python files: non-test source, string literals
(raw, byte and `\\`-continued Rust strings; Python f-strings, concatenations and
adjacent literals), public surfaces (Rust `pub` items and methods, pyo3 exports,
Python `__all__`), assertions in cited tests, and `.charge(` inside loops.
They report facts; the checker decides which fact is a violation.
"""

import ast
import hashlib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from _repo import REPO_ROOT
from _py_closure import closure_nodes
from _source_cache import source_cache
from test_evidence import (
    _attrs_before,
    _is_cfg_test,
    _match_bracket,
    _rust_closure_parts,
    closure,
    rust_items,
    target_modules,
)

# ------------------------------------------------------------- non-test source


def crate_src(path: Path) -> Path | None:
    for parent in path.parents:
        if parent.name == "src":
            return parent
    return None


@source_cache
def _compiled_test_only(path: Path) -> bool | None:
    """True if every target compiling `path` does so only under cfg(test); None if
    no target of its crate compiles it."""
    src = crate_src(path)
    if src is None:
        return None
    merged: dict[Path, bool] = {}
    roots = [src / "lib.rs", src / "main.rs", *sorted((src / "bin").glob("*.rs"))]
    for top in roots:
        if top.is_file():
            for mod, test_only in target_modules(top.resolve()).items():
                merged[mod] = merged.get(mod, True) and test_only
    return merged.get(path.resolve())


@source_cache


def nontest_pair(path: Path, *, in_crate: bool = True) -> tuple[str, str]:
    """(code, skeleton) of `path` with comments removed and #[cfg(test)]
    modules/functions and #[test] functions blanked; both empty when the file is
    compiled only for tests or not compiled at all. `code` keeps string literals,
    `skeleton` blanks their contents; offsets agree."""
    key = path.resolve()
    out = ("", "")
    compiled = _compiled_test_only(path) is False if in_crate else True
    if compiled:
        masked, items = rust_items(key)
        if not re.search(r"#!\[cfg\(\s*test\s*\)\]", masked.code):
            code, skel = list(masked.code), list(masked.skeleton)
            for item in items:
                if item.kind not in ("fn", "mod"):
                    continue
                if _is_cfg_test(item.attrs) or any(
                    re.sub(r"\s+", "", a) == "#[test]" for a in item.attrs
                ):
                    for k in range(item.start, item.body[1]):
                        if code[k] != "\n":
                            code[k] = skel[k] = " "
            out = ("".join(code), "".join(skel))
    return out


def nontest_rust(path: Path, *, in_crate: bool = True) -> str:
    return nontest_pair(path, in_crate=in_crate)[0]


# --------------------------------------------------------------------- literals


@dataclass(frozen=True)
class Lit:
    line: int
    value: str
    const: str | None  # name of the const/static the literal initialises, if any
    reason_code: str | None = None  # proven closed helper constructor, otherwise unknown


# A char literal (so `'"'` is not a string start), a raw string opener, or a
# (byte) string. A backslash-newline continuation is joined below.
_LIT = re.compile(
    r"""'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]{1,6}\}|.)|[^\\'\n])'"""
    r"""|(?<![\w])b?r(?P<hash>#*)\""""
    r"""|(?<![\w])b?"(?P<str>(?:\\.|[^"\\])*)\"""",
    re.DOTALL,
)


def _scan_literals(code: str, lo: int = 0, hi: int | None = None) -> list[tuple[int, str]]:
    """(offset, value) of every string literal in comment-free `code[lo:hi]`:
    plain, byte, raw (`r"…"`, `r#"…"#`) and `\\`-continued (joined to one value)."""
    hi = len(code) if hi is None else hi
    out: list[tuple[int, str]] = []
    i = lo
    while True:
        m = _LIT.search(code, i, hi)
        if not m:
            break
        if m.group("hash") is not None:
            hashes = m.group("hash")
            end = code.find('"' + hashes, m.end(), hi)
            end = hi if end == -1 else end
            value, i = code[m.end() : end], end + 1 + len(hashes)
        elif m.group("str") is not None:
            value, i = re.sub(r"\\\r?\n\s*", "", m.group("str")), m.end()
        else:
            i = m.end()
            continue
        out.append((m.start(), value))
    return out


def _compact_rust_tokens(code: str) -> str:
    """Remove inter-token whitespace while preserving complete literal tokens.

    Proof comparisons use this instead of deleting all whitespace: spaces inside
    string and character literals are part of the emitted value and must remain
    observable to the source proof.
    """
    pieces: list[str] = []
    cursor = 0
    while match := _LIT.search(code, cursor):
        pieces.append(re.sub(r"\s+", "", code[cursor : match.start()]))
        if match.group("hash") is not None:
            hashes = match.group("hash")
            close = code.find('"' + hashes, match.end())
            end = len(code) if close < 0 else close + 1 + len(hashes)
        else:
            end = match.end()
        pieces.append(code[match.start() : end])
        cursor = end
    pieces.append(re.sub(r"\s+", "", code[cursor:]))
    return "".join(pieces)


def _core_refusal_helpers() -> dict[str, str]:
    """Prove the exact code/detail forwarding of core's two artifact helpers."""
    path = REPO_ROOT / "crates/antecedent-core/src/refusal.rs"
    code, skel = nontest_pair(path)
    _, items = rust_items(path.resolve())
    expected = {
        "changed_identity": (
            'fnchanged_identity(detail:&str,field:implfmt::Display)->RefusalParts',
            '(crate::reason_code!("route_not_supported"),detail.to_owned(),format!("{field} changed"))',
            "route_not_supported",
        ),
        "replay_mismatch": (
            'fnreplay_mismatch(detail:&str,explanation:String)->RefusalParts',
            '(crate::reason_code!("invalid_argument"),detail.to_owned(),explanation)',
            "invalid_argument",
        ),
    }
    found = {}
    for name, (header, body, reason) in expected.items():
        defs = [item for item in items if item.kind == "fn" and item.name == name]
        if len(defs) != 1:
            continue
        item = defs[0]
        if (re.search(r"\bpub\s*$", skel[:item.start])
                and _compact_rust_tokens(code[item.start:item.body[0]]) == header
                and _compact_rust_tokens(code[item.body[0]:item.body[1]]) == "{" + body + "}"):
            found[name] = reason
    return found


def _imported_core_refusal_helpers(skel: str) -> set[str]:
    imported: set[str] = set()
    for match in re.finditer(r"\buse\s+antecedent_core::refusal::([^;]+);", skel):
        names = match[1].strip().removeprefix("{").removesuffix("}")
        imported.update(name.strip() for name in names.split(",")
                        if name.strip() in {"changed_identity", "replay_mismatch"})
    # Shadowing or a local definition invalidates an unqualified-call proof.
    return {name for name in imported if not re.search(
        rf"\b(?:fn|let|const|static|struct|enum|type)\s+{name}\b", skel)}


def _multiple_imputation_error_helpers() -> dict[str, str]:
    """Prove this sibling type's exact code and detail forwarding."""
    path = REPO_ROOT / "crates/antecedent-estimate/src/multiple_imputation/mod.rs"
    code, skel = nontest_pair(path)
    block = re.search(r"\bimpl\s+MultipleImputationError\s*\{", skel)
    if block is None:
        return {}
    opening = block.end() - 1
    closing = _match_bracket(skel, opening)
    _, items = rust_items(path.resolve())
    expected = {
        "new": ("fnnew(code:&'staticstr,detail:&'staticstr,message:implInto<String>,next:Option<NextAttempt>,)->Self",
                "{letmessage=message.into();letmessage=matchnext{Some(attempt)=>attempt.append_to(&message),None=>message,};Self{code,detail,message,next}}"),
        "invalid": ("fninvalid(detail:&'staticstr,message:implInto<String>)->Self",
                    '{Self::new(antecedent_core::reason_code!("invalid_argument"),detail,message,None)}'),
        "not_applicable": ("fnnot_applicable(detail:&'staticstr,message:implInto<String>,next:NextAttempt,)->Self",
                           '{Self::new(antecedent_core::reason_code!("missing_policy_not_applicable"),detail,message,Some(next),)}'),
    }
    valid = set()
    for name, (header, body) in expected.items():
        matches = [item for item in items if item.kind == "fn" and item.name == name
                   and opening < item.start < closing]
        if len(matches) == 1:
            item = matches[0]
            if (_compact_rust_tokens(code[item.start:item.body[0]]) == header
                    and _compact_rust_tokens(code[item.body[0]:item.body[1]]) == body):
                valid.add(name)
    if "new" not in valid:
        return {}
    return {name: reason for name, reason in (("invalid", "invalid_argument"),
            ("not_applicable", "missing_policy_not_applicable")) if name in valid}


def _horizon_refusal_is_proven(path: Path, code: str, skel: str) -> bool:
    """Prove the exact horizon helper and transitive ExternalRefusal setters."""
    expected_path = REPO_ROOT / "crates/antecedent/src/analysis/horizon_composition.rs"
    if path.resolve() != expected_path.resolve():
        return False
    if not re.search(r"\buse\s+antecedent_contract::\{[^}]*\bExternalRefusal\b[^}]*\}\s*;", skel):
        return False
    library = REPO_ROOT / "crates/antecedent-contract/src/lib.rs"
    _, lib_skel = nontest_pair(library)
    if not re.search(r"\bpub\s+use\s+external_refusal::\{[^}]*\bExternalRefusal\b[^}]*\}\s*;", lib_skel):
        return False
    dependencies = [
        (path, code, {
            "refuse": ("fnrefuse(code:&'staticstr,detail:&str,offending:implInto<String>,remedy:&'staticstr,)->ExternalRefusal",
                       "{ExternalRefusal::new(code,JOINT_LAW_STAGE,detail).with_offending(offending).with_remedy(remedy)}"),
        }),
    ]
    external = REPO_ROOT / "crates/antecedent-contract/src/external_refusal.rs"
    external_code, external_skel = nontest_pair(external)
    impl_match = re.search(r"\bimpl\s+ExternalRefusal\s*\{", external_skel)
    if impl_match is None:
        return False
    lo = impl_match.end() - 1
    hi = _match_bracket(external_skel, lo)
    _, external_items = rust_items(external.resolve())
    expected = {
        "new": ("fnnew(code:&'staticstr,stage:&'staticstr,detail:implInto<String>)->Self",
                "{Self{code,stage,detail:detail.into(),offending:None,expected:None,supplied:None,capability:None,remedy:None,}}"),
        "with_offending": ("fnwith_offending(mutself,value:implInto<String>)->Self",
                           "{self.offending=Some(value.into());self}"),
        "with_remedy": ("fnwith_remedy(mutself,value:&'staticstr)->Self",
                        "{self.remedy=Some(value);self}"),
    }
    for name, (header, body) in expected.items():
        found = [item for item in external_items if item.kind == "fn" and item.name == name and lo < item.start < hi]
        if (len(found) != 1 or _compact_rust_tokens(external_code[found[0].start:found[0].body[0]]) != header
                or _compact_rust_tokens(external_code[found[0].body[0]:found[0].body[1]]) != body):
            return False
    _, items = rust_items(path.resolve())
    found = [item for item in items if item.kind == "fn" and item.name == "refuse"]
    header, body = dependencies[0][2]["refuse"]
    return (len(found) == 1 and _compact_rust_tokens(code[found[0].start:found[0].body[0]]) == header
            and _compact_rust_tokens(code[found[0].body[0]:found[0].body[1]]) == body)


def _source_item_hash(path: Path, kind: str, name: str) -> str | None:
    """Fingerprint one non-test function, const or inherent impl after token normalization."""
    code, skel = nontest_pair(path)
    if kind == "impl":
        found = list(re.finditer(rf"\bimpl\s+{re.escape(name)}\s*\{{", skel))
        if len(found) != 1:
            return None
        start = found[0].start()
        end = _match_bracket(skel, found[0].end() - 1) + 1
    else:
        _, items = rust_items(path.resolve())
        found = [item for item in items if item.kind == kind and item.name == name]
        if len(found) != 1:
            return None
        start, end = found[0].start, found[0].body[1]
    return hashlib.sha256(_compact_rust_tokens(code[start:end]).encode()).hexdigest()


def _adaptive_premise_proven(path: Path) -> bool:
    """Prove Error::PremiseFailed survives the facade with cell_not_licensed."""
    base = REPO_ROOT / "crates/antecedent-stats/src/adaptive_inference"
    producers = {
        (base / "repeated_measures_posterior.rs").resolve(): ("premise_failure", "ee58cf70efbfd761a85d19d29808869bc0a9b7fae688d46821fff0c3c2811479"),
        (base / "procedure_uncertainty.rs").resolve(): ("premise", "ebde83de6f9c7b4948a7585661eef4cf33efa8faeae63a6006fcc5bd59cb099c"),
        (REPO_ROOT / "crates/antecedent/src/analysis/adaptive_procedure_report.rs").resolve(): ("refusal_of", "cc7d439d6ef611d653d461b664ca078fc0da34963fc271daef2da48f6d302142"),
    }
    producer = producers.get(path.resolve())
    if producer is None:
        return False
    _, producer_skel = nontest_pair(path)
    if (path.name != "adaptive_procedure_report.rs"
            and (not re.search(r"\buse\s+super::AdaptiveInferenceError\s+as\s+Error\s*;", producer_skel)
                 or re.search(r"\b(?:type|struct|enum|trait)\s+Error\b", producer_skel))):
        return False
    dependencies = [
        (path, "fn", producer[0], producer[1]),
        (base / "procedure_uncertainty.rs", "fn", "premise_reason_code", "42797da7b36544a2f2eacdf8a1356f85bd860f1286091cdd7d2d3fcb7fce3b61"),
        (REPO_ROOT / "crates/antecedent/src/analysis/adaptive_procedure_report.rs", "fn", "refusal_of", "cc7d439d6ef611d653d461b664ca078fc0da34963fc271daef2da48f6d302142"),
        (REPO_ROOT / "crates/antecedent/src/analysis/adaptive_procedure_report.rs", "fn", "premise_remedy", "1abc9c49543a905f0e7e393e417fb1c67d66f2d01f601f8ff2bfcb6fc3083f94"),
        (REPO_ROOT / "crates/antecedent/src/analysis/adaptive_procedure_report.rs", "const", "ASSUMPTION_DETAILS", "96cc07be8e8d97a6edbf81e75420dc2c649792377f90019d2db0f434bf5bcb01"),
    ]
    if path.name == "repeated_measures_posterior.rs":
        dependencies.append((REPO_ROOT / "crates/antecedent/src/analysis/repeated_measures_posterior_report.rs", "fn", "compute", "633c763abdd67b1601e015594186bf784df2c0299958934d0377c571d485a0ba"))
    return all(_source_item_hash(source, kind, name) == expected for source, kind, name, expected in dependencies)


def _adaptive_rule_emitter(path: Path, skel: str) -> str | None:
    """Prove exact rule-mismatch constructors without guessing a default code."""
    rel = path.resolve().relative_to(REPO_ROOT.resolve()) if path.resolve().is_relative_to(REPO_ROOT.resolve()) else None
    rel = str(rel) if rel else ""
    stats = "crates/antecedent-stats/src/adaptive_inference/"
    gap_sources = {stats + "procedure_contract.rs", stats + "procedure_repeated_measures.rs", stats + "participant_procedure.rs"}
    if rel in gap_sources:
        gap = REPO_ROOT / (stats + "procedure_contract.rs")
        if _source_item_hash(gap, "impl", "ContractGap") != "fb66b233f6acc63eecf90e70a4dc8669d1fa207a100d4fcf98774742525b1a44":
            return None
        if rel != stats + "procedure_contract.rs" and not re.search(r"\buse\s+super::procedure_contract::\{[^}]*\bContractGap\b[^}]*\}\s*;", skel):
            return None
        if rel != stats + "procedure_contract.rs" and re.search(r"\b(?:struct|enum|type|trait)\s+ContractGap\b", skel):
            return None
        return "gap"
    contracts = {
        "crates/antecedent-contract/src/adaptive_design/declaration.rs": [
            ("declare", "f50c26f68b03b7967607d6033ef386da2d464a0fe50b102d414407b74e99945a"),
            ("invalid", "80208575687d2300970c5b134d0b83ae1a77c62ded70bbf4b359ab95589d4c81")],
        "crates/antecedent-io-core/src/adaptive_monitoring_contract_artifact.rs": [
            ("procedure_refusal", "b26ed0b0349dc851bc05f504a7e1356b0247f9cf6e8c7f1dbd9ebef67fe16a2c")],
        "crates/antecedent/src/analysis/adaptive_procedure_declaration.rs": [
            ("refusal", "3c7d90b4dd9904c4ead13696460f020c932b5ad18ede2250668e436085e142ca"),
            ("invalid", "bb030285d2ea89a4953ce8cc76b2ca7815ba48d0e0aa6fa4a90312a8114e3316")],
    }
    funcs = contracts.get(rel)
    if funcs is None:
        return None
    # `declare` uses this concrete external refusal constructor and setter.
    if rel.startswith("crates/antecedent-contract/"):
        external = REPO_ROOT / "crates/antecedent-contract/src/external_refusal.rs"
        if _source_item_hash(external, "impl", "ExternalRefusal") != "f34fc42e5d96be3c45bcbba2133c59856e967f51e90e4b5b2cef9d7c1355c6d5":
            return None
    if not all(_source_item_hash(path, "fn", name) == digest for name, digest in funcs):
        return None
    return "procedure_refusal" if rel.startswith("crates/antecedent-io-core/") else "invalid"


def _recalc_no_state_proven(path: Path, skel: str) -> bool:
    """Prove the concrete Python bridge forwards the NoLiveState code and detail."""
    base = REPO_ROOT / "python/src"
    approved = {(base / name).resolve() for name in (
        "recalc_api.rs", "recalc_adjusted_api.rs", "recalc_cell_api.rs", "recalc_design_api.rs",
        "recalc_dr_api.rs", "recalc_family_api.rs", "recalc_static_api.rs")}
    if path.resolve() not in approved:
        return False
    if path.name != "recalc_api.rs":
        if not re.search(r"\buse\s+crate::recalc_api::(?:refusal_json|\{[^}]*\brefusal_json\b[^}]*\})\s*;", skel):
            return False
        if re.search(r"\bfn\s+refusal_json\b", skel):
            return False
    dependencies = [
        (base / "recalc_api.rs", "fn", "refusal_body", "12a8db04302b9c4c4ff0ee079e8984d85bf688bca0c17d788bd00c9aafe7dbff"),
        (base / "recalc_api.rs", "fn", "refusal_json", "488f335dc50a5edb1ee1ce692668c60b9a42fd52db1d5395ed88eb24005da178"),
        (base / "bridge.rs", "impl", "RefusalJson", "cb37996f75876d30ef8f1ae67abe7f31b5e42d158cce990044576c75f6be08e9"),
    ]
    return all(_source_item_hash(source, kind, name) == digest for source, kind, name, digest in dependencies)


def _decision_builder_proven(path: Path) -> bool:
    """Prove decision-reliability's fixed builder and completed refusal arms."""
    expected = REPO_ROOT / "crates/antecedent-design/src/decision_reliability.rs"
    if path.resolve() != expected.resolve():
        return False
    dependencies = [
        (expected, "fn", "builder", "eccc9fbabd0196489c8cb02b7164dbf40290bef9244cb623eae4c40b59182657"),
        (expected, "fn", "to_refusal", "91f41ccd81fbb6c68f6bdfd71614d1c1f72bd14aca5bab4ca7787007ced7c409"),
        (REPO_ROOT / "crates/antecedent-design/src/signal.rs", "impl", "RefusalBuilder", "948280df6891ab404eefc1886763c6f4e99950209e04485b6214b936c55fcfbc"),
        (REPO_ROOT / "crates/antecedent-contract/src/external_refusal.rs", "impl", "ExternalRefusal", "f34fc42e5d96be3c45bcbba2133c59856e967f51e90e4b5b2cef9d7c1355c6d5"),
    ]
    return all(_source_item_hash(source, kind, name) == digest for source, kind, name, digest in dependencies)


def rust_literals(path: Path, *, in_crate: bool = True) -> list[Lit]:
    """Every string literal in non-test Rust source, with the const it initialises."""
    code, skel = nontest_pair(path, in_crate=in_crate)
    if not code:
        return []
    _, items = rust_items(path.resolve())
    consts = [it for it in items if it.kind == "const"]
    out: list[Lit] = []
    provider_literals: set[int] = set()
    imported = _imported_core_refusal_helpers(skel)
    core_helpers = _core_refusal_helpers() if imported else {}
    horizon_proven = _horizon_refusal_is_proven(path, code, skel) if path.name == "horizon_composition.rs" else False
    premise_proven = _adaptive_premise_proven(path) if path.name in {"procedure_uncertainty.rs", "repeated_measures_posterior.rs", "adaptive_procedure_report.rs"} else False
    adaptive_rule = _adaptive_rule_emitter(path, skel) if "adaptive_inference.rule_mismatch" in code else None
    recalc_no_state = _recalc_no_state_proven(path, skel) if "recalc.no_live_state" in code else False
    decision_builder = _decision_builder_proven(path) if path.name == "decision_reliability.rs" else False
    mi_dir = REPO_ROOT / "crates/antecedent-estimate/src/multiple_imputation"
    mi_helpers = {}
    if (path.resolve() in {(mi_dir / "impute.rs").resolve(), (mi_dir / "pool.rs").resolve()}
            and re.search(r"\buse\s+super::(?:MultipleImputationError|\{[^}]*\bMultipleImputationError\b[^}]*\})\s*;", skel)
            and not re.search(r"\b(?:struct|enum|type|trait)\s+MultipleImputationError\b", skel)
            and not re.search(r"\buse\s+(?!super::)[^;]*\bMultipleImputationError\b", skel)):
        mi_helpers = _multiple_imputation_error_helpers()
    for block in re.finditer(r"\bDistributionProvenance\s*\{", skel):
        opening = block.end() - 1
        closing = _match_bracket(skel, opening)
        for match in re.finditer(r'provider_id\s*:\s*"[^"\n]*"\.to_owned\(\)\s*,', code[opening + 1:closing]):
            start = opening + 1 + match.start()
            prefix = skel[opening + 1:start]
            if (re.match(r"provider_id\s*:", skel[start:])
                    and prefix.count("{") == prefix.count("}")
                    and prefix.count("(") == prefix.count(")")):
                provider_literals.add(code.index('"', start, opening + 1 + match.end()))
    remedy_ranges: list[tuple[int, int]] = []
    # This exact report lookup is descriptive only while the full premise
    # forwarding proof, including its body fingerprint, remains valid.
    adaptive_report = REPO_ROOT / "crates/antecedent/src/analysis/adaptive_procedure_report.rs"
    if premise_proven and path.resolve() == adaptive_report.resolve():
        remedy_ranges.extend(item.body for item in items
                             if item.kind == "fn" and item.name == "premise_remedy")
    for item in items:
        if (item.kind == "fn" and item.name == "remedy_for"
                and re.fullmatch(r"fn\s+remedy_for\s*\(\s*detail\s*:\s*&str\s*\)\s*->\s*&'static\s+str\s*", code[item.start:item.body[0]])):
            remedy_ranges.append(item.body)
    for match in re.finditer(r"\blet\s+remedy\s*=\s*match\s+detail\.as_str\(\)\s*\{", skel):
        opening = match.end() - 1
        remedy_ranges.append((opening, _match_bracket(skel, opening)))
    scanned = _scan_literals(code)
    # The generated artifact enum's identity label and display text are not
    # independent reason-code emitters. Exclude only metadata for a detail whose
    # actual direct core-helper call has already been proven in this same file.
    proven_details: set[str] = set()
    for at, value in scanned:
        for helper in imported:
            if (helper in core_helpers and re.search(
                    rf"(?<![\w.:]){helper}\s*\(\s*$", skel[:at])):
                proven_details.add(value)
    enum_metadata: set[int] = set()
    for macro in re.finditer(r"\bcrate::sectioned_artifact_enum!\s*\{", skel):
        opening = macro.end() - 1
        closing = _match_bracket(skel, opening)
        displays = []
        for display in re.finditer(r"\bdisplay\s*\(\s*f\s*\)\s*\{", skel[opening + 1:closing]):
            begin = opening + 1 + display.end() - 1
            displays.append((begin, _match_bracket(skel, begin)))
        for at, value in scanned:
            if not opening < at < closing:
                continue
            detail = value.split(":", 1)[0]
            if detail not in proven_details:
                continue
            before = skel[opening + 1:at]
            if value == detail and re.search(r"\bidentity\s*:\s*$", before):
                enum_metadata.add(at)
            elif (any(lo < at < hi for lo, hi in displays)
                  and re.search(r"\bwrite!\s*\(\s*f\s*,\s*$", before)):
                enum_metadata.add(at)
    for at, value in scanned:
        # A match-pattern in a recognized remedy-only lookup is descriptive.
        # Other match patterns can map a detail to a code and remain scanned.
        token = _LIT.match(code, at)
        match_pattern = bool(token and token.group("str") is not None
                             and _inside(remedy_ranges, at)
                             and re.match(r"\s*=>", skel[token.end():]))
        if at in provider_literals or at in enum_metadata or match_pattern:
            continue
        # A sentence appended to report limits is explanatory text, not a
        # dynamically constructed refusal detail. Restrict this to the literal
        # argument of limits.push(format!(...)); error constructors stay scanned.
        if (path.name == "report.rs"
                and value == "{} directions are left unresolved by the recovery."
                and re.search(r"\blimits\s*\.\s*push\s*\(\s*format!\s*\(\s*$", skel[:at])):
            continue
        const = next((c.name for c in consts if c.body[0] <= at < c.body[1]), None)
        reason = None
        premise_details = {"adaptive_inference.selection_not_ignorable", "adaptive_inference.arm_not_observed",
                           "adaptive_inference.structure_incomplete", "adaptive_inference.rule_mismatch"}
        if (premise_proven and value in premise_details
                and (re.search(r"(?<![\w.:])(?:premise_failure|premise)\s*\(\s*$", skel[:at])
                     or (path.name == "adaptive_procedure_report.rs" and re.search(
                         r"\brefusal_of\s*\(\s*&AdaptiveInferenceError::PremiseFailed\s*\{\s*detail\s*:\s*$", skel[:at])))):
            reason = "cell_not_licensed"
        if decision_builder and value in {"decision_reliability.search_record_required",
                                          "decision_reliability.validation_not_independent",
                                          "decision_reliability.dependence_unknown"}:
            built = re.search(r'(?<![\w.:])builder\s*\(\s*antecedent_core::reason_code!\(\s*"([a-z][a-z0-9_]*)"\s*\)\s*,\s*$', code[:at])
            if built and re.match(r"builder\s*\(", skel[built.start():]):
                reason = built[1]
        if value == "recalc.no_live_state" and recalc_no_state:
            direct_no_state = re.search(
                r'(?<![\w.:])refusal_json\s*\(\s*antecedent_core::reason_code!\(\s*"score_table_unavailable"\s*\)\s*,\s*&stage\.label\(\)\s*,\s*$', code[:at])
            if direct_no_state and re.match(r"refusal_json\s*\(", skel[direct_no_state.start():]):
                reason = "score_table_unavailable"
        if value == "adaptive_inference.rule_mismatch" and adaptive_rule:
            prefix = (r"(?<![\w:])ContractGap::Invalid\s*\(\s*$" if adaptive_rule == "gap"
                      else rf"(?<![\w.:]){adaptive_rule}\s*\(\s*$")
            if re.search(prefix, skel[:at]):
                reason = "invalid_argument"
        # Recognize the literal reason-code/detail pair at a direct local
        # refuse call. The skeleton check prevents matching inside a string or
        # comment; unsupported constructor shapes remain unproven.
        direct = re.search(
            r'(?<![\w.:])refuse\s*\(\s*(?:antecedent_core::)?reason_code!\(\s*"([a-z][a-z0-9_]*)"\s*\)\s*,\s*$',
            code[:at],
        )
        if horizon_proven and direct and re.match(r"refuse\s*\(", skel[direct.start():]):
            reason = direct[1]
        for helper, proven_code in mi_helpers.items():
            if re.search(rf"(?<![\w:])MultipleImputationError::{helper}\s*\(\s*$", skel[:at]):
                reason = proven_code
                break
        for helper in imported:
            if helper in core_helpers and re.search(
                rf"(?<![\w.:]){helper}\s*\(\s*$", skel[:at]
            ):
                reason = core_helpers[helper]
                break
        out.append(Lit(code.count("\n", 0, at) + 1, value, const, reason))
    return out


def refusal_stage_literals(path: Path, *, in_crate: bool = True) -> set[tuple[int, str]]:
    """Only exact RefusalFields.stage values are stage metadata, not reason details."""
    code, skel = nontest_pair(path, in_crate=in_crate)
    output = set()
    pattern = re.compile(
        r'stage\s*:\s*Some\(\s*"([a-z][a-z0-9_]*\.[a-z][a-z0-9_]*)"\.to_owned\(\)\s*\)\s*,'
    )
    for block in re.finditer(r"\bRefusalFields\s*\{", skel):
        opening = block.end() - 1
        closing = _match_bracket(skel, opening)
        for match in pattern.finditer(code, opening + 1, closing):
            # Only direct struct members qualify, never nested messages/calls.
            prefix = skel[opening + 1 : match.start()]
            if prefix.count("{") == prefix.count("}") and prefix.count("(") == prefix.count(")"):
                line = code.count("\n", 0, match.start()) + 1
                if (
                    sum(
                        lit.line == line and lit.value == match[1]
                        for lit in rust_literals(path, in_crate=in_crate)
                    )
                    == 1
                ):
                    output.add((line, match[1]))
    return output


def finite_refusal_helpers(path: Path, *, in_crate: bool = True) -> tuple[list[Lit], set[int]]:
    """Expand a private leaf-module formatter only when every suffix is literal.

    This is a proof over the closed call set, not a waiver for dynamic strings.
    Unrecognized bodies, escaping function pointers and public/child-module
    access leave the original dynamic literal for the checker to reject.
    """
    code, skel = nontest_pair(path, in_crate=in_crate)
    if path.name in {"lib.rs", "main.rs", "mod.rs"} or re.search(
        r"\bmod\s+[A-Za-z_]\w*\s*[;{]|\binclude\s*!|\bmacro_rules\s*!", skel
    ):
        return [], set()
    _, items = rust_items(path.resolve())
    output: list[Lit] = []
    proven_lines: set[int] = set()
    literals = rust_literals(path, in_crate=in_crate)
    macro_ranges = []
    for macro in re.finditer(r"\b\w+(?:::\w+)*\s*!\s*([({\[])", skel):
        opening = macro.end() - 1
        macro_ranges.append((opening, _match_bracket(skel, opening)))
    body_pattern = re.compile(
        r"\s*EstimationError::(?P<constructor>refused|refused_with_fields)\s*\(\s*"
        r'antecedent_core::reason_code!\(\s*"(?P<code>[a-z][a-z0-9_]+)"\s*\)\s*,\s*'
        r'format!\(\s*"(?P<namespace>[a-z][a-z0-9_]*)\.\{detail\}: \{message\}"\s*\)\s*,'
        r"\s*(?P<fields>fields\s*,\s*)?\)\s*",
        re.DOTALL,
    )
    structured_body_pattern = re.compile(
        r"\s*EstimationError::(?P<constructor>refused_detail|refused_detail_with_fields)\s*\(\s*"
        r'antecedent_core::reason_code!\(\s*"(?P<code>[a-z][a-z0-9_]+)"\s*\)\s*,\s*'
        r'format!\(\s*"(?P<namespace>[a-z][a-z0-9_]*)\.\{detail\}"\s*\)\s*,\s*message\s*,'
        r"\s*(?P<fields>fields\s*,\s*)?\)\s*",
        re.DOTALL,
    )
    for item in items:
        if item.kind != "fn" or not skel[item.start : item.body[0]].strip():
            continue
        header = code[item.start : item.body[0]]
        if not re.fullmatch(
            rf"fn\s+{re.escape(item.name)}\s*\(\s*detail\s*:\s*&str\s*,\s*"
            r"message\s*:\s*&str\s*(?:,\s*fields\s*:\s*RefusalFields\s*)?\)"
            r"\s*->\s*EstimationError\s*",
            header,
        ):
            continue
        prefix = skel[: item.start]
        if prefix.count("{") != prefix.count("}") or re.search(
            r"\b(?:pub|async|unsafe|extern)(?:\([^)]*\))?\s*$", prefix
        ):
            continue
        # The parser's body range includes braces; no extra work or delegation
        # is admitted in this narrowly recognized formatter.
        body = code[item.body[0] : item.body[1]].strip().removeprefix("{").removesuffix("}")
        matched = body_pattern.fullmatch(body) or structured_body_pattern.fullmatch(body)
        if not matched or (
            (matched["constructor"] in {"refused_with_fields", "refused_detail_with_fields"}) != bool(matched["fields"])
        ):
            continue
        if ("fields" in header) != bool(matched["fields"]):
            continue
        uses = []
        valid = True
        for use in re.finditer(rf"\b{re.escape(item.name)}\b", skel):
            if item.start <= use.start() < item.body[0]:
                continue
            # Qualified/member calls, recursion, passing the helper as a value,
            # and a non-literal or composed first argument are not finite proof.
            tail = code[use.end() :]
            argument = re.match(r'\s*\(\s*"([a-z][a-z0-9_]*)"\s*,', tail)
            if (
                not argument
                or (item.body[0] <= use.start() < item.body[1])
                or (use.start() and skel[use.start() - 1] in ".:")
                or any(lo <= use.start() < hi for lo, hi in macro_ranges)
            ):
                valid = False
                break
            uses.append(
                Lit(
                    code.count("\n", 0, use.start()) + 1,
                    matched["namespace"] + "." + argument[1],
                    None,
                    matched["code"],
                )
            )
        if valid and uses:
            output.extend(uses)
            proven_lines.update(
                lit.line
                for lit in literals
                if lit.value == matched["namespace"] + (
                    ".{detail}" if matched["constructor"].startswith("refused_detail")
                    else ".{detail}: {message}"
                )
                and code.count("\n", 0, item.body[0]) + 1
                <= lit.line
                <= code.count("\n", 0, item.body[1]) + 1
                and sum(other.line == lit.line and other.value == lit.value for other in literals)
                == 1
            )
    # This private leaf formatter takes a literal reason-code macro and a
    # literal scope at every call. No aliases, escapes, recursion, qualified
    # calls or unrecognized arguments are admitted by the closed-call proof.
    for item in items:
        if item.kind != "fn" or item.name != "transfer_refusal":
            continue
        header = _compact_rust_tokens(code[item.start:item.body[0]])
        body = _compact_rust_tokens(code[item.body[0]:item.body[1]])
        if (header != "fntransfer_refusal(code:&'staticstr,scope:&str,message:&str)->CausalError"
                or body != '{antecedent_io::IoError::Refused{code,message:format!("mapped_posterior_transfer.{scope}: {message}"),}.into()}'):
            continue
        prefix = skel[:item.start]
        if prefix.count("{") != prefix.count("}") or re.search(
            r"\b(?:pub|async|unsafe|extern)(?:\([^)]*\))?\s*$", prefix
        ):
            continue
        uses = []
        valid = True
        for use in re.finditer(r"\btransfer_refusal\b", skel):
            if item.start <= use.start() < item.body[0]:
                continue
            argument = re.match(
                r'\s*\(\s*antecedent_core::reason_code!\(\s*"([a-z][a-z0-9_]*)"\s*\)\s*,\s*"([a-z][a-z0-9_]*)"\s*,',
                code[use.end():],
            )
            if (not argument or item.body[0] <= use.start() < item.body[1]
                    or (use.start() and skel[use.start() - 1] in ".:")
                    or any(lo <= use.start() < hi for lo, hi in macro_ranges)):
                valid = False
                break
            uses.append(Lit(code.count("\n", 0, use.start()) + 1,
                            "mapped_posterior_transfer." + argument[2], None, argument[1]))
        dynamic = "mapped_posterior_transfer.{scope}: {message}"
        if valid and uses:
            output.extend(uses)
            proven_lines.update(
                lit.line for lit in literals
                if lit.value == dynamic
                and code.count("\n", 0, item.body[0]) + 1 <= lit.line
                <= code.count("\n", 0, item.body[1]) + 1
                and sum(other.line == lit.line and other.value == lit.value for other in literals) == 1
            )
    output.extend(sectioned_artifact_refusals(path, in_crate=in_crate))
    return output, proven_lines


# This is the deliberately closed set of production invocations of
# `sectioned_artifact_error!`. A new generated refusal namespace must be
# reviewed here as well as registered in promotion.toml; arbitrary macro calls
# or strings cannot create evidence for an unreviewed namespace.
_SECTIONED_ARTIFACTS = {
    "crates/antecedent-io/src/effect_constancy_artifact.rs": (
        "effect_constancy",
        False,
    ),
    "crates/antecedent-io/src/vector_treatment_artifact.rs": (
        "vector_treatment",
        False,
    ),
    "crates/antecedent-io/src/categorical_treatment_artifact.rs": (
        "categorical_treatment",
        False,
    ),
    "crates/antecedent-io/src/nonlinear_mediation_artifact.rs": (
        "nonlinear_mediation",
        True,
    ),
    "crates/antecedent-io/src/latent_class_artifact.rs": ("latent_class", True),
    "crates/antecedent-io/src/mechanism_discrepancy_artifact.rs": (
        "mechanism_discrepancy",
        False,
    ),
    "crates/antecedent-io/src/predictive_assessment_artifact.rs": (
        "predictive_assessment",
        True,
    ),
}


def _sectioned_error_macro_is_proven() -> bool:
    """Recognize only the current macro arms that construct registered details."""
    macro_path = (
        Path(__file__).resolve().parents[1] / "crates/antecedent-io-core/src/artifact_codec.rs"
    )
    code, skel = nontest_pair(macro_path)
    _, items = rust_items(macro_path)
    definitions = [
        item for item in items if item.kind == "macro" and item.name == "sectioned_artifact_error"
    ]
    if len(definitions) != 1:
        return False
    item = definitions[0]
    if [re.sub(r"\s+", "", attr) for attr in _attrs_before(skel, code, item.start)] != [
        "#[macro_export]"
    ]:
        return False
    methods = [
        method
        for method in items
        if method.kind == "fn"
        and method.name == "refusal"
        and item.body[0] < method.start < item.body[1]
    ]
    if len(methods) != 1:
        return False
    method = methods[0]
    header = _compact_rust_tokens(code[method.start - 4 : method.body[0]])
    body = _compact_rust_tokens(code[method.body[0] : method.body[1]])
    expected_header = "pubfnrefusal(&self)->Option<(&'staticstr,String,String)>"
    expected_body = (
        "{matchself{Self::Refused{code,message}=>{let(detail,text)="
        'message.split_once(": ").unwrap_or((message.as_str(),""));'
        "Some((*code,detail.to_owned(),text.to_owned()))}"
        'Self::IdentityMismatch{field}=>Some((antecedent_core::reason_code!("route_not_supported"),'
        'format!("{contract}.wrong_contract",contract=$contract),'
        'format!("{field} changed"),)),'
        'Self::ResultMismatch(what)=>{if$report{Some((antecedent_core::reason_code!("invalid_argument"),'
        'format!("{contract}.report_replay_mismatch",contract=$contract),'
        'format!("the stored {what} does not replay"),))}else{None}}_=>None,}}'
    )
    structured_body = (
        "{matchself{Self::Refused{code,detail,explanation}=>{Some((*code,detail.clone(),explanation.clone()))}"
        'Self::IdentityMismatch{field}=>{Some(antecedent_core::refusal::changed_identity(&format!("{contract}.wrong_contract",contract=$contract),field,))}'
        'Self::ResultMismatch(what)=>{if$report{Some(antecedent_core::refusal::replay_mismatch(&format!("{contract}.report_replay_mismatch",contract=$contract),format!("the stored {what} does not replay"),))}else{None}}_=>None,}}'
    )
    if header != expected_header:
        return False
    if body == expected_body:
        return True
    if body != structured_body:
        return False
    helper_path = macro_path.parents[2] / "antecedent-core/src/refusal.rs"
    helper_code, _ = nontest_pair(helper_path)
    _, helper_items = rust_items(helper_path)
    helpers = {
        method.name: _compact_rust_tokens(helper_code[method.body[0]:method.body[1]])
        for method in helper_items
        if method.kind == "fn" and method.name in {"changed_identity", "replay_mismatch"}
    }
    return helpers == {
        "changed_identity": '{(crate::reason_code!("route_not_supported"),detail.to_owned(),format!("{field} changed"))}',
        "replay_mismatch": '{(crate::reason_code!("invalid_argument"),detail.to_owned(),explanation)}',
    }


def sectioned_artifact_refusals(path: Path, *, in_crate: bool = True) -> list[Lit]:
    """Expand the closed production table of generated artifact refusals.

    The macro definition is checked for the actual reason-code and detail
    expressions. At each reviewed call site the namespace and report flag must
    still match the finite table above. Only live invocation tokens are parsed;
    strings and comments are blanked by the Rust lexer.
    """
    repo = Path(__file__).resolve().parents[1]
    try:
        relative = path.resolve().relative_to(repo).as_posix()
    except ValueError:
        return []
    expected = _SECTIONED_ARTIFACTS.get(relative)
    if expected is None or not _sectioned_error_macro_is_proven():
        return []
    code, skel = nontest_pair(path, in_crate=in_crate)
    calls = list(re.finditer(r"(?<![\w:])crate::sectioned_artifact_error\s*!\s*([({\[])", skel))
    if len(calls) != 1:
        return []
    if any(
        re.sub(r"\s+", "", attr).startswith("#[cfg")
        for attr in _attrs_before(skel, code, calls[0].start())
    ):
        return []
    opening = calls[0].end() - 1
    closing = _match_bracket(skel, opening)
    # Split only on commas at this invocation's top level. Literal contents
    # and nested attributes are blank in `skel`, so their punctuation cannot
    # manufacture arguments.
    parts: list[tuple[int, int]] = []
    start = opening + 1
    depth = 0
    for index in range(start, closing):
        char = skel[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append((start, index))
            start = index + 1
    if code[start:closing].strip():
        parts.append((start, closing))
    if len(parts) != 5:
        return []
    args = [code[lo:hi].strip() for lo, hi in parts]
    name = re.fullmatch(r"[A-Z][A-Za-z0-9_]*", args[0])
    literals = []
    for arg in args[1:4]:
        match = re.fullmatch(r'"([a-z][a-z0-9_ ]*)"', arg)
        if not match:
            return []
        literals.append(match[1])
    report = args[4]
    if report not in {"true", "false"} or not name:
        return []
    _noun, contract, _replay = literals
    if (contract, report == "true") != expected:
        return []
    line = code.count("\n", 0, calls[0].start()) + 1
    emitted = [Lit(line, f"{contract}.wrong_contract", None, "route_not_supported")]
    if report == "true":
        emitted.append(Lit(line, f"{contract}.report_replay_mismatch", None, "invalid_argument"))
    return emitted


_RECOVERY_EXTERNAL_SYMBOLS = frozenset({
    "CertifiedInterventionEvaluation", "CertifiedInterventionEvaluation.outcome",
    "CertifiedInterventionEvaluation.draws", "CertifiedInterventionEvaluation.total_mass",
    "CertifiedInterventionEvaluation.withheld_mass", "CertifiedInterventionEvaluation.outcome_mixture",
    "RecoveryEvidenceDecision", "RecoveryEvidenceDecision.evaluation",
    "RecoveryEvidenceDecision.above_threshold", "RecoveryEvidenceDecision.at_or_below_threshold",
    "RecoveryEvidenceDecision.withheld_mass", "RecoveryEvidenceDecision.choice",
    "RecoveryEvidenceDecision.mass_account", "consume_retained_recovery_plan",
    "consume_recovery_evidence", "asymptotic_normal_draws",
})


def external_surface_ownership(root: Path, record: dict, checked: list[tuple[str, "Symbol"]]):
    """Validate the finite cyclic/recovery ownership handoff; never grant a license."""
    entries = record.get("surface_external_owners", [])
    if not isinstance(entries, list):
        return set(), ["surface_external_owners must be a list"]
    if not entries:
        return set(), []
    expected = ("recovery.binary_observation", "parity/cyclic/acceptance.toml",
                "integrated_families", "crates/antecedent-estimate/src/recovery.rs")
    if len(entries) != 1 or not isinstance(entries[0], dict):
        return set(), ["external ownership requires one reviewed declaration"]
    entry = entries[0]
    if set(entry) != {"registry", "owner", "source", "symbols"}:
        return set(), ["external ownership needs exactly registry, owner, source and symbols"]
    if (record.get("id"), entry["registry"], entry["owner"], entry["source"]) != expected:
        return set(), ["external ownership is not the reviewed cyclic recovery handoff"]
    symbols = entry["symbols"]
    if (not isinstance(symbols, list) or not all(isinstance(s, str) and s for s in symbols)
            or len(symbols) != len(set(symbols)) or set(symbols) != _RECOVERY_EXTERNAL_SYMBOLS):
        return set(), ["external ownership must name exactly the reviewed, unique recovery symbols"]
    if entry["source"] not in (record.get("surface_rust") or []):
        return set(), ["external ownership source is not a declared Rust surface"]
    registry = (root / expected[1]).resolve()
    source = (root / expected[3]).resolve()
    if not registry.is_relative_to(root.resolve()) or not source.is_relative_to(root.resolve()):
        return set(), ["external ownership paths escape the repository"]
    try:
        document = tomllib.loads(registry.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        return set(), ["external ownership registry cannot be read"]
    capabilities = document.get("capability", [])
    if not isinstance(capabilities, list):
        return set(), ["external ownership registry has no capability inventory"]
    matches = [c for c in capabilities if isinstance(c, dict) and c.get("id") == expected[2]]
    if len(matches) != 1:
        return set(), ["external ownership requires a unique existing capability"]
    owner = matches[0]
    rust = owner.get("rust")
    workflow = owner.get("workflow")
    tests, fixtures = owner.get("tests"), owner.get("fixtures")
    if (not isinstance(rust, list) or expected[3] not in rust
            or not isinstance(workflow, str) or "prepare_experimental_recovery" not in workflow
            or not isinstance(tests, list) or not tests
            or "crates/antecedent-estimate/tests/recovery_held_out_regimes.rs" not in tests
            or not isinstance(fixtures, list) or not fixtures
            or "parity/cyclic/recovered-asymptotic-normal" not in fixtures):
        return set(), ["external owner lacks the recovery source, workflow, tests or fixtures"]
    for name in [*tests, *fixtures]:
        if not isinstance(name, str) or not name or Path(name).is_absolute():
            return set(), ["external owner has an invalid evidence path"]
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.exists():
            return set(), ["external owner has missing or escaping evidence"]
    available: dict[str, int] = {}
    for src, symbol in checked:
        if src == expected[3]:
            qualified = f"{symbol.owner}.{symbol.name}" if symbol.owner else symbol.name
            available[qualified] = available.get(qualified, 0) + 1
    if any(available.get(name) != 1 for name in symbols):
        return set(), ["external ownership has a missing, stale or ambiguous public symbol"]
    return {(expected[3], name) for name in symbols}, []


def duplicate_refusal_pairs(refusals: list[dict]) -> set[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    duplicates: set[tuple[str, str]] = set()
    for refusal in refusals:
        pair = (refusal.get("code", ""), refusal.get("detail", ""))
        if pair in seen:
            duplicates.add(pair)
        seen.add(pair)
    return duplicates


def refusal_code_pair_problems(declared: set[str], registered: set[str],
                               proven: set[str]) -> tuple[set[str], set[str]]:
    """Return unproven declared codes and proven but globally unregistered codes."""
    require_proof = bool(proven) or len(registered) > 1
    return (declared - proven if require_proof else set(), proven - registered)


# The only reviewed early uncertainty exception is this observation-query redraw path.
# The facade, dispatcher, producer, bindings and the two refusal tests were reviewed together.
# Hashing their complete bytes makes any source or test change fail closed until re-reviewed.
_OBSERVATION_DRAWS_PROOF_FILES = {
    "crates/antecedent-model/src/observation_uncertainty.rs": "5e86d83476f0085f61a3b31a26569f5e7adb27ea1fbf522d0176541b8ddaf82b",
    "crates/antecedent-io-core/src/observation_query_artifact.rs": "b445b4df8fd45b26d16586437d145cd8d0f1911c8b1cf318f29d835ca89b6b23",
    "crates/antecedent/src/analysis/observation_query.rs": "f59b27fc750bf311f5efdb00308b3234f68bd9e3b577635b6c302768c5530876",
    "crates/antecedent/tests/observation_query_facade.rs": "24b99901b7b173382d8cd04d1d2927a3e8a0bc8022f065ea9baea892b5bbcccb",
    "python/src/observation_query_api.rs": "211ab22c82a88cd92eecb83ab9f92391aec7b48603f67f2c1a0b40a7fdb0d7e7",
    "python/antecedent/observation_query.py": "e82024841c481484c9d49bc498923efdc3b1e750e50fb2c05e1d1163f61cbacc",
    "python/tests/test_observation_query.py": "9ad244fa91540c4c086b31efb9624b0f4caf69831fab921e3ee0f5ee026c092b",
}
_OBSERVATION_DRAWS_RUST = (
    "antecedent.observation_query.ObservationClaim.parameter_interval",
    "crates/antecedent/tests/observation_query_facade.rs",
    "dirichlet_table_redraw_agrees_with_the_count_fitted_posterior_and_labels_its_spread",
)
_OBSERVATION_DRAWS_PYTHON = (
    "antecedent._native.observation_query_parameter_interval",
    "python/tests/test_observation_query.py",
    "test_an_interval_of_redrawn_parameters_needs_a_measured_coverage_record",
)
_OBSERVATION_DRAWS_ROUTES = {
    "antecedent_model.observation_uncertainty.redraw_conditional": _OBSERVATION_DRAWS_RUST,
    "antecedent.observation_query.PreparedMechanismComposition.observe_with_parameter_uncertainty": _OBSERVATION_DRAWS_RUST,
    "antecedent.python.observation_query.parameter_uncertainty": _OBSERVATION_DRAWS_PYTHON,
    "antecedent._native.observation_query_observe_with_uncertainty": _OBSERVATION_DRAWS_PYTHON,
}


def uncalibrated_draws_allowed(record_id: str, route: dict, routes: list[dict], valid_refusal,
                               *, root: Path = REPO_ROOT) -> bool:
    """Accept only the reviewed draw producers and their exercised interval refusals."""
    if (record_id != "observation_query.finite_law_conditioning"
            or route.get("stage") != "uncertainty"
            or route.get("status") != "licensed"
            or route.get("claim") != "point_only"
            or route.get("uncertainty_output") != "uncalibrated_parameter_draws"):
        return False
    proof = _OBSERVATION_DRAWS_ROUTES.get(route.get("name"))
    if proof is None or route.get("interval_route") != proof[0]:
        return False
    matches = [r for r in routes if isinstance(r, dict) and r.get("name") == proof[0]]
    if len(matches) != 1:
        return False
    interval = matches[0]
    if (interval.get("stage") != "uncertainty" or interval.get("status") != "closed"
            or interval.get("reason_code") != "cell_not_licensed"
            or interval.get("refusal_test") != proof[1]
            or interval.get("refusal_assertion") != proof[2]):
        return False
    root = root.resolve()
    try:
        for relative, expected in _OBSERVATION_DRAWS_PROOF_FILES.items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                return False
    except OSError:
        return False
    return bool(valid_refusal(interval))


def shared_namespace_problems(
    record: dict, records: list[dict], emissions: dict, root: Path
) -> list[str]:
    """Prove finite shared namespace ownership, never blanket multi-prefix permission."""
    problems = []
    declarations = record.get("shared_refusal_namespaces", [])
    if not isinstance(declarations, list):
        return ["shared_refusal_namespaces must be a list"]
    namespaces = {r.get("detail", "").split(".")[0] for r in record.get("refusals", [])}
    seen = set()
    for declaration in declarations:
        if not isinstance(declaration, dict) or set(declaration) != {
            "namespace",
            "owner_record",
            "sources",
        }:
            problems.append("shared namespace needs exactly namespace, owner_record, sources")
            continue
        ns, owner, sources = (declaration[k] for k in ("namespace", "owner_record", "sources"))
        if (
            not isinstance(ns, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]*", ns)
            or ns in seen
            or ns not in namespaces
        ):
            problems.append("shared namespace must be unique and actually declared")
            continue
        seen.add(ns)
        if (
            not isinstance(sources, list)
            or not sources
            or any(not isinstance(p, str) for p in sources)
            or len(set(sources)) != len(sources)
        ):
            problems.append(f"{ns}: sources must be a finite unique nonempty list")
            continue
        actual = {
            str(path.resolve().relative_to(root.resolve()))
            if path.resolve().is_relative_to(root.resolve())
            else str(path)
            for uses in emissions.get(ns, {}).values()
            for path, _, _ in uses
        }
        if actual != set(sources) or any(
            not p.endswith(".rs") or "/src/" not in p or ".." in Path(p).parts for p in sources
        ):
            problems.append(f"{ns}: sources must exactly equal emitting production Rust files")
        owners = [r for r in records if r.get("id") == owner]
        if len(owners) != 1 or owners[0].get("status") not in {
            "promoted",
            "in_progress",
        }:
            problems.append(f"{ns}: missing implemented namespace owner")
            continue
        if declaration not in owners[0].get("shared_refusal_namespaces", []):
            problems.append(f"{ns}: owner must declare identical sources and self ownership")
        owned = {(r.get("code"), r.get("detail")) for r in owners[0].get("refusals", [])}
        if (
            any(
                (r.get("code"), r.get("detail")) not in owned
                for r in record.get("refusals", [])
                if r.get("detail", "").startswith(ns + ".")
            )
            or not set(emissions.get(ns, {})) <= {detail for _, detail in owned}
            or any((lit.reason_code, detail) not in owned
                   for detail, uses in emissions.get(ns, {}).items()
                   for _, _, lit in uses if getattr(lit, "reason_code", None) is not None)
        ):
            problems.append(f"{ns}: every emitted refusal pair must be declared by owner")
    if len(namespaces - seen) > 1:
        problems.append("refusal details must share one primary namespace")
    return problems


def defensive_carrier_route_problems(root: Path) -> list[str]:
    """Factory and fresh consumer permit exactly the declared source-report routes."""
    native, native_skel = nontest_pair(root / "python/src/measured_inference_api.rs")
    io, _ = nontest_pair(root / "crates/antecedent-io/src/measured_inference.rs")
    expected = {
        "joint_bayesian",
        "learned_joint",
        "nested_fisher",
        "nested_bayesian",
        "sampled_recovery",
        "temporal_interval",
        "adaptive_interval",
        "cohort_interval",
        "count_fitted_interval",
        "sampled_counterfactual_transport",
        "split_effect",
        "split_parent_set_effect",
    }
    _, native_items = rust_items(root / "python/src/measured_inference_api.rs")
    _, io_items = rust_items(root / "crates/antecedent-io/src/measured_inference.rs")
    source = [
        native[i.body[0] : i.body[1]]
        for i in native_items
        if i.kind == "fn" and i.name == "source_report"
    ]
    consume = [
        io[i.body[0] : i.body[1]] for i in io_items if i.kind == "fn" and i.name == "consume"
    ]

    def arms(text: str) -> set[str]:
        return set(re.findall(r'"([a-z_]+)"\s*=>', text))

    if (
        len(source) != 1
        or len(consume) != 1
        or arms(source[0]) != expected
        or arms(consume[0]) != expected
        or "#[new]" in native_skel
        or "#[setter" in native_skel
        or '_ => Err(refusal("measured_inference.unsupported_route"' not in consume[0]
        or '_ => Err(crate::value_err("measured_inference.source_report_not_supported"))'
        not in source[0]
    ):
        return [
            "defensive source-only claim requires the exact declared factory/fresh-consumer route match and no public constructor/setter"
        ]
    # Every native IO factory call uses a literal member of the same finite
    # route set. Unknown/dynamic factory arguments and escaped aliases reject
    # the defensive-only claim even when the consumer still rejects them.
    factory_routes = set()
    constructor_count = 0
    for path in (root / "crates/antecedent-io/src").glob("**/*.rs"):
        code, skeleton = nontest_pair(path)
        constructor_count += len(re.findall(r"\bMeasuredInference\s*\{", skeleton)) - len(
            re.findall(r"\b(?:struct|impl)\s+MeasuredInference\s*\{", skeleton)
        )
        if re.search(r"\bauthorize\s+as\b", skeleton):
            return ["defensive factory proof cannot permit an escaped authority helper alias"]
        for imported in reversed(list(re.finditer(r"\buse\b[^;]+;", skeleton))):
            lo, hi = imported.span()
            code = code[:lo] + " " * (hi - lo) + code[hi:]
            skeleton = skeleton[:lo] + " " * (hi - lo) + skeleton[hi:]
        for call in re.finditer(r"\bauthorize\b", skeleton):
            if re.search(r"\bfn\s*$", skeleton[: call.start()]):
                continue
            argument = re.match(r'\s*\(\s*"([a-z_]+)"\s*,', code[call.end() :])
            if not argument or argument[1] not in expected:
                return [
                    "defensive factory proof requires a finite literal route at every native authority call"
                ]
            factory_routes.add(argument[1])
    if factory_routes != expected or constructor_count != 1:
        return [
            "defensive factory proof requires exactly the declared routes and one private authority constructor"
        ]
    # Native carrier and IO authority cannot deserialize or expose mutable
    # report/source fields. The checked authority bypasses this route match.
    if (
        not re.search(
            r"pub\(crate\) struct NativeMeasuredInference\s*\{\s*authority: NativeAuthority,\s*payload: String,\s*source_report: OnceLock<Arc<str>>,\s*\}",
            native,
        )
        or not re.search(
            r"pub struct MeasuredInference\s*\{\s*report: MeasuredInferenceReport,\s*bytes: Arc<\[u8\]>,\s*source: Arc<\[u8\]>,\s*\}",
            io,
        )
        or re.search(r"#\[derive\([^]]*Deserialize[^]]*\)\]\s*pub struct MeasuredInference\b", io)
    ):
        return ["defensive carrier route must remain opaque, immutable and non-deserializable"]
    return []


def input_exception_evidence_problems(entry: dict, path: Path, assertion: str) -> list[str]:
    """Require class assertion plus absent reason code, including finite pytest cases."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == assertion
    ]
    if len(functions) != 1:
        return ["class-only evidence must name one ordinary test"]
    fn = functions[0]
    if entry.get("defensive_source_only") is True:
        # Runtime status is never inferred from this static defensive test.
        text = ast.unparse(fn)
        return (
            []
            if entry["detail"] in text and entry["helper"] in text and "assert" in text
            else ["defensive evidence must assert the exact guarded source helper"]
        )
    assertions = "\n".join(ast.unparse(n) for n in ast.walk(fn) if isinstance(n, ast.Assert))
    raises = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "pytest.raises"
    ]
    if "reason_code" not in assertions or "None" not in assertions:
        return ["class-only runtime evidence must assert absent reason_code"]
    class_name = entry["exception_class"]
    if any(n.args and ast.unparse(n.args[0]) == class_name for n in raises):
        return []
    # Explicit literal pytest parameter tuple must tie detail and class to the
    # same exercised case; unrelated names elsewhere in the file prove nothing.
    for decorator in fn.decorator_list:
        if (
            not isinstance(decorator, ast.Call)
            or ast.unparse(decorator.func) != "pytest.mark.parametrize"
            or len(decorator.args) < 2
        ):
            continue
        try:
            names, values = (
                ast.literal_eval(decorator.args[0]),
                ast.literal_eval(decorator.args[1]),
            )
        except (ValueError, TypeError):
            continue
        if isinstance(names, str):
            names = [name.strip() for name in names.split(",")]
        if (
            not isinstance(names, (tuple, list))
            or "exception" not in names
            or "detail" not in names
        ):
            continue
        if not any(
            n.args and ast.unparse(n.args[0]) == "getattr(errors, exception)" for n in raises
        ):
            continue
        if any(
            isinstance(row, (tuple, list))
            and len(row) == len(names)
            and row[names.index("exception")] == class_name
            and row[names.index("detail")] == entry["detail"].split(".")[1]
            for row in values
        ):
            return []
    return ["class-only runtime evidence must exercise the exact declared class/detail case"]


def typed_input_exception_problems(entry: dict, root: Path) -> list[str]:
    """Class-only native precondition errors; scientific refusals never qualify."""
    allowed = {
        "measured_inference.invalid_expectation": (
            "CausalValueError",
            "crate::value_err",
        ),
        "measured_inference.source_report_not_supported": (
            "CausalValueError",
            "crate::value_err",
        ),
        "measured_inference.consumer_bounds_exceeded": (
            "CausalResourceError",
            "crate::CausalResourceError::new_err",
        ),
        "measured_inference.memory_budget_exceeded": (
            "CausalResourceError",
            "crate::CausalResourceError::new_err",
        ),
    }
    required = {
        "detail",
        "exception_class",
        "helper",
        "source",
        "reason_code_absent",
        "evidence_test",
        "evidence_assertion",
    }
    if (
        not isinstance(entry, dict)
        or not required <= entry.keys()
        or entry.keys() - required - {"defensive_source_only"}
    ):
        return [
            "class-only declaration needs the exact proof fields and cannot declare a reason code or unknown fields"
        ]
    detail = entry.get("detail", "")
    pair = (entry.get("exception_class"), entry.get("helper"))
    source = "python/src/measured_inference_api.rs"
    if (
        detail not in allowed
        or pair != allowed[detail]
        or entry.get("source") != source
        or entry.get("reason_code_absent") is not True
    ):
        return [
            "class-only declaration must match an actual bounded native precondition/resource error"
        ]
    defensive = entry.get("defensive_source_only", False)
    if defensive and detail != "measured_inference.source_report_not_supported":
        return ["live precondition/resource branch cannot be declared defensive_source_only"]
    if detail == "measured_inference.source_report_not_supported":
        if defensive is not True:
            return ["unreachable source-report guard must be explicitly defensive_source_only"]
        problems = defensive_carrier_route_problems(root)
        if problems:
            return problems
    code, skel = nontest_pair(root / source)
    helper = re.escape(str(entry["helper"]))
    literal = re.escape(detail)
    call = re.compile(helper + r'\s*\(\s*(?:format!\(\s*)?"' + literal + r'(?::[^"\n]*)?"')
    matches = list(call.finditer(code))
    if not matches:
        return ["declared helper does not emit this literal in non-test source"]
    occurrences = [
        offset
        for offset, value in _scan_literals(code)
        if value == detail or value.startswith(detail + ":")
    ]
    if any(
        not any(match.start() <= offset < match.end() for match in matches)
        for offset in occurrences
    ):
        return ["this class-only detail is also emitted outside its proven typed helper"]
    for wrapper in re.finditer(r"\bwith_reason_code\s*\(", skel):
        close = _match_bracket(skel, wrapper.end() - 1)
        if any(wrapper.end() <= match.start() < close for match in matches):
            return ["class-only constructor is wrapped in a reason-code assignment"]
    # The registered helper may change; prove the actual current mapping rather
    # than equating arbitrary functions with ValueError by their spelling.
    if pair[0] == "CausalValueError":
        error_source = root / "python/src/lib.rs"
        lib, lib_skeleton = nontest_pair(error_source)
        _, items = rust_items(error_source)
        if not any(item.kind == "fn" and item.name == "value_err" for item in items):
            if not re.search(r"\bmod\s+errors\s*;", lib_skeleton) or not re.search(
                r"\bpub\(crate\)\s+use\s+errors::\*\s*;", lib_skeleton
            ):
                return ["value_err helper module lacks its checked crate reexport"]
            error_source = root / "python/src/errors.rs"
            lib, _ = nontest_pair(error_source)
            _, items = rust_items(error_source)
        bodies = [
            lib[i.body[0] : i.body[1]] for i in items if i.kind == "fn" and i.name == "value_err"
        ]
        body = _compact_rust_tokens(bodies[0]) if len(bodies) == 1 else ""
        if (
            len(bodies) != 1
            or body
            != "{letmessage=message.as_ref();attach_reason_from_message(value_error_of(message),message)}"
        ):
            return [
                "value_err no longer proves registered value-error mapping without a reason code"
            ]
        value_errors = [
            lib[item.body[0] : item.body[1]]
            for item in items
            if item.kind == "fn" and item.name == "value_error_of"
        ]
        value_error_body = _compact_rust_tokens(value_errors[0]) if len(value_errors) == 1 else ""
        if (
            len(value_errors) != 1
            or value_error_body
            != "{Python::attach(|py|{VALUE_ERROR_CLASS.get(py).and_then(|cls|cls.bind(py).call1((message,)).ok()).map_or_else(||CausalValidateError::new_err(message.to_string()),PyErr::from_value)})}"
        ):
            return ["value_err no longer proves its registered CausalValueError mapping"]
        reason_helpers = {
            item.name: _compact_rust_tokens(lib[item.body[0] : item.body[1]])
            for item in items
            if item.kind == "fn"
            and item.name in {"reason_code_in_message", "attach_reason_from_message"}
        }
        if reason_helpers != {
            "reason_code_in_message": '{antecedent_core::reason_code::split_prefix(message).or_else(||{message.split_once(": ").and_then(|(_,rest)|antecedent_core::reason_code::split_prefix(rest))}).map(|(code,_)|code)}',
            "attach_reason_from_message": "{matchreason_code_in_message(message).filter(|code|antecedent_core::reason_code::is_registered(code)){Some(code)=>with_reason_code(err,code),None=>err,}}",
        }:
            return [
                "value_err reason propagation no longer proves class-only literal messages remain uncoded"
            ]
        reason_path = root / "crates/antecedent-core/src/reason_code.rs"
        reason_code, reason_skel = nontest_pair(reason_path)
        _, reason_items = rust_items(reason_path)
        split_prefix = [
            _compact_rust_tokens(reason_code[item.body[0] : item.body[1]])
            for item in reason_items
            if item.kind == "fn" and item.name == "split_prefix"
        ]
        prefix_items = [
            item for item in reason_items if item.kind == "const" and item.name == "PREFIX"
        ]
        prefix_declarations = list(re.finditer(r"(?<![\w])pub\s+const\s+PREFIX\b", reason_skel))
        prefix_is_exact = False
        if len(prefix_items) == len(prefix_declarations) == 1:
            prefix_item = prefix_items[0]
            prefix_start = prefix_declarations[0].start()
            prefix_is_exact = (
                prefix_start < prefix_item.start < prefix_item.body[1]
                and not any(
                    fn.body[0] < prefix_start < fn.body[1] for fn in reason_items if fn.kind == "fn"
                )
                and _compact_rust_tokens(reason_code[prefix_start : prefix_item.body[1]])
                == 'pubconstPREFIX:&str=concat!("reason","=");'
            )
        if (
            len(split_prefix) != 1
            or split_prefix[0]
            != "{letrest=message.strip_prefix(PREFIX)?;let(code,tail)=rest.split_once(':')?;Some((code.trim(),tail.trim_start()))}"
            or not prefix_is_exact
        ):
            return ["registered reason prefix parser changed; class-only mapping needs review"]
        if any(
            "reason=" in value
            for match in matches
            for _, value in _scan_literals(code, match.start(), match.end())
        ):
            return ["class-only value_err literal now includes an explicit reason prefix"]
    if pair[0] == "CausalValueError":
        errors = ast.parse((root / "python/antecedent/errors.py").read_text(encoding="utf-8"))
        registrations = [
            n
            for n in ast.walk(errors)
            if isinstance(n, ast.Call)
            and ast.unparse(n.func) == "_set_value_error_class"
            and [ast.unparse(arg) for arg in n.args] == ["CausalValueError"]
        ]
        classes = [
            n
            for n in ast.walk(errors)
            if isinstance(n, ast.ClassDef)
            and n.name == "CausalValueError"
            and [ast.unparse(base) for base in n.bases] == ["CausalValidateError", "ValueError"]
        ]
        if len(registrations) != 1 or len(classes) != 1:
            return ["registered value-error class mapping no longer matches the native helper"]
    return []


def ignored_test_literals(path: Path) -> dict[str, list[str]]:
    """Test name -> string literals in the body of every `#[ignore]`d fn of a Rust
    test file, comments excluded (the coverage-record ids a calibration test emits)."""
    masked, items = rust_items(path.resolve())
    out: dict[str, list[str]] = {}
    for it in items:
        if it.kind != "fn" or not any(
            re.sub(r"\s+", "", a).startswith("#[ignore") for a in it.attrs
        ):
            continue
        lo, hi = it.body
        out.setdefault(it.name, []).extend(
            value for _, value in _scan_literals(masked.code, lo, hi)
        )
    for name, values in _macro_generated_ignored_tests(masked, items).items():
        out.setdefault(name, []).extend(values)
    return out


def _macro_generated_ignored_tests(masked, items) -> dict[str, list[str]]:
    """Resolve ignored test generators, including single-arm forwarding macros.

    Only actual invocations outside macro definitions count. Forwarders must be
    one macro call, with fixed, named parameters; arguments are substituted as
    Rust tokens before following the call. The generated test gets literals and
    named constant tables from the arguments actually reaching its generator.
    Unused wrapper arguments and unrelated label tables cannot provide evidence.
    """
    from _rust_source import lex

    skel, code = masked.skeleton, masked.code
    constants = {it.name: it for it in items if it.kind == "const"}
    macros = {it.name: it for it in items if it.kind == "macro"}
    definitions = {}

    def arguments(text: str) -> list[str]:
        skeleton = lex(text).skeleton
        result, start, index = [], 0, 0
        while index < len(skeleton):
            char = skeleton[index]
            if char in "([{":
                index = _match_bracket(skeleton, index) + 1
                continue
            if char == ",":
                result.append(text[start:index].strip())
                start = index + 1
            index += 1
        tail = text[start:].strip()
        if tail:
            result.append(tail)
        return result

    for name, macro in macros.items():
        text = code[macro.body[0] : macro.body[1]]
        skeleton = lex(text).skeleton
        opening = skeleton.find("(")
        if opening < 0:
            continue
        closing = _match_bracket(skeleton, opening)
        params = arguments(text[opening + 1 : closing])
        names = []
        for param in params:
            match = re.fullmatch(r"\$(\w+)\s*:\s*(?:ident|expr|literal|path|ty|tt)", param)
            if match is None:
                break
            names.append(match[1])
        if len(names) != len(params) or not names or len(set(names)) != len(names):
            continue
        arm = re.match(r"\s*=>\s*\{", skeleton[closing + 1 :])
        if arm is None:
            continue
        body_start = closing + 1 + arm.end() - 1
        body_end = _match_bracket(skeleton, body_start)
        if not re.fullmatch(r"\s*;?\s*\}\s*", skeleton[body_end + 1 :]):
            continue  # Multiple arms/repetition need a separate explicit parser.
        body = text[body_start + 1 : body_end]
        body_skeleton = skeleton[body_start + 1 : body_end]
        generated_names = re.findall(r"\bfn\s+\$(\w+)\s*\(", body_skeleton)
        if (
            "#[ignore" in re.sub(r"\s+", "", body_skeleton)
            and len(generated_names) == 1
            and generated_names[0] in names
        ):
            definitions[name] = (names, generated_names[0], None)
            continue
        call = re.match(r"\s*(\w+)\s*!\s*[(\[{]", body_skeleton)
        if call is None:
            continue
        end = _match_bracket(body_skeleton, call.end() - 1)
        if re.fullmatch(r"\s*;?\s*", body_skeleton[end + 1 :]):
            definitions[name] = (names, None, (call[1], body[call.end() : end]))

    def resolve(name: str, text: str, seen: frozenset[str]):
        if name in seen or name not in definitions:
            return None
        params, generated, forwarded = definitions[name]
        supplied = arguments(text)
        if len(supplied) != len(params):
            return None
        bindings = dict(zip(params, supplied, strict=True))
        if generated is not None:
            test = bindings[generated]
            return (test, text) if re.fullmatch(r"[A-Za-z_]\w*", test) else None
        target, forwarded_text = forwarded
        tokens = list(re.finditer(r"\$(\w+)", lex(forwarded_text).skeleton))
        for token in reversed(tokens):
            if token[1] not in bindings:
                return None
            forwarded_text = (
                forwarded_text[: token.start()] + bindings[token[1]] + forwarded_text[token.end() :]
            )
        return resolve(target, forwarded_text, seen | {name})

    # Preserve the existing direct-generator literal audit for repetition and
    # multi-arm definitions. These do not gain forwarding semantics implicitly.
    legacy_generators = {
        name
        for name, macro in macros.items()
        if name not in definitions
        and "#[ignore" in re.sub(r"\s+", "", code[macro.body[0] : macro.body[1]])
        and re.search(r"\bfn\s+\$\w+", code[macro.body[0] : macro.body[1]])
    }
    out: dict[str, list[str]] = {}
    for name in definitions:
        call = re.compile(r"(?<![\w$])" + re.escape(name) + r"\s*!\s*[(\[{]")
        for match in call.finditer(skel):
            if any(m.body[0] <= match.start() < m.body[1] for m in macros.values()):
                continue
            end = _match_bracket(skel, match.end() - 1)
            resolved = resolve(name, code[match.end() : end], frozenset())
            if resolved is None:
                continue
            test, actual = resolved
            actual_source = lex(actual)
            values = [test] + [v for _, v in _scan_literals(actual_source.kept)]
            for ident in set(re.findall(r"[A-Za-z_]\w*", actual_source.skeleton)):
                const = constants.get(ident)
                if const is not None:
                    values += [v for _, v in _scan_literals(code, const.body[0], const.body[1])]
            out.setdefault(test, []).extend(values)
    for name in legacy_generators:
        call = re.compile(r"(?<![\w$])" + re.escape(name) + r"\s*!\s*[(\[{]")
        for match in call.finditer(skel):
            if any(m.body[0] <= match.start() < m.body[1] for m in macros.values()):
                continue
            end = _match_bracket(skel, match.end() - 1)
            actual = code[match.end() : end]
            first = re.match(r"\s*([A-Za-z_]\w*)", actual)
            if first is None:
                continue
            values = [first[1]] + [v for _, v in _scan_literals(actual)]
            for ident in set(re.findall(r"[A-Za-z_]\w*", lex(actual).skeleton)):
                const = constants.get(ident)
                if const is not None:
                    values += [v for _, v in _scan_literals(code, const.body[0], const.body[1])]
            out.setdefault(first[1], []).extend(values)
    return out


def ignored_test_emits_record(path: Path, name: str) -> bool:
    """Whether the ignored test `name` can emit a coverage record: a test that scores only
    through `CoverageTally::new` (a pass/fail gate, no `for_record` key) emits none."""
    masked, items = rust_items(path.resolve())
    for it in items:
        if it.kind == "fn" and it.name == name:
            body = masked.code[it.body[0] : it.body[1]]
            if (
                "CoverageTally::new(" in body
                and "for_record" not in body
                and "RecordKey" not in body
            ):
                return False
    return True


def python_fn_literals(path: Path) -> dict[str, list[str]]:
    """Function name -> string constants in its body, for every def of a Python file."""
    try:
        tree = ast.parse(path.read_text(errors="ignore", encoding="utf-8"))
    except SyntaxError:
        return {}
    out: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.setdefault(node.name, []).extend(
                sub.value
                for stmt in node.body
                for sub in ast.walk(stmt)
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
            )
    return out


def _py_str(node: ast.AST) -> str | None:
    """The string an expression builds, with `{}` for every non-constant part
    (f-string field, `+` operand, `%` argument); None if it is not a string."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return node.value
        if isinstance(node.value, bytes):
            return node.value.decode("utf-8", "ignore")
        return None
    if isinstance(node, ast.JoinedStr):
        return "".join(
            v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else "{}"
            for v in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _py_str(node.left), _py_str(node.right)
        if left is None and right is None:
            return None
        return (left if left is not None else "{}") + (right if right is not None else "{}")
    return None


def python_literals(path: Path) -> list[Lit]:
    """Every string a Python file builds: constants (adjacent literals are already
    one constant), f-strings and `+` concatenations."""
    try:
        tree = ast.parse(path.read_text(errors="ignore", encoding="utf-8"))
    except SyntaxError:
        return []
    # Documentation is not an emitted runtime string. In particular, a sentence
    # ending in a namespace word such as "scenarios." is not a dynamic refusal.
    # Keep ordinary constants and actual f-strings/concatenations fully scanned.
    docstrings: set[int] = set()
    for scope in ast.walk(tree):
        if isinstance(scope, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = scope.body[0] if scope.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                docstrings.add(id(first.value))
    # A literal proof description on this exact decorator is metadata, not
    # an emitted refusal. Do not exclude expressions, other keywords, regular
    # function calls, or equal strings elsewhere in the file.
    artifact_proofs: set[int] = set()
    for scope in ast.walk(tree):
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in scope.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                func = decorator.func
                if not (isinstance(func, ast.Name) and func.id == "artifact_consumer"):
                    continue
                for keyword in decorator.keywords:
                    if (keyword.arg == "proof" and isinstance(keyword.value, ast.Constant)
                            and isinstance(keyword.value.value, str)):
                        artifact_proofs.add(id(keyword.value))
    # These two reviewed Python functions only classify an already-produced
    # refusal by its detail. Their exact bodies must still match before their
    # comparison literals can be treated as non-emitting metadata.
    classification_specs = {
        (REPO_ROOT / "python/antecedent/recalc.py").resolve():
            ("_refusal_type", "19fd58bfde934b9d9d29a2f24622ad5e798249d3079dc4f7a701c52d1dc0cdb2"),
        (REPO_ROOT / "python/antecedent/recalc_cell.py").resolve():
            ("_cell_refusal_type", "29ba42a5038a58e4b3e05cb02900118801833d1f2422bf1e89edefd02aa6d85c"),
    }
    classification_literals: set[int] = set()
    specification = classification_specs.get(path.resolve())
    if specification:
        name, expected = specification
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
        if len(functions) == 1 and hashlib.sha256(
            ast.dump(functions[0], include_attributes=False).encode()
        ).hexdigest() == expected:
            for node in ast.walk(functions[0]):
                if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
                        and node.left.id == "detail" and len(node.ops) == 1
                        and isinstance(node.ops[0], ast.Eq) and len(node.comparators) == 1
                        and isinstance(node.comparators[0], ast.Constant)
                        and node.comparators[0].value == "recalc.no_live_state"):
                    classification_literals.add(id(node.comparators[0]))
    consts: dict[int, str] = {}
    for node in ast.walk(tree):
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        value = getattr(node, "value", None)
        if value is not None and any(isinstance(t, ast.Name) for t in targets):
            name = next(t.id for t in targets if isinstance(t, ast.Name))
            for sub in ast.walk(value):
                consts.setdefault(id(sub), name)
    seen: set[tuple[int, str]] = set()
    out: list[Lit] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.Constant, ast.JoinedStr, ast.BinOp))
            and id(node) not in docstrings
            and id(node) not in artifact_proofs
            and id(node) not in classification_literals
        ):
            value = _py_str(node)
            if value is not None and (node.lineno, value) not in seen:
                seen.add((node.lineno, value))
                out.append(Lit(node.lineno, value, consts.get(id(node))))
    return out


# --------------------------------------------------------------- public surface


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: str  # "type" | "function" | "other"
    hidden: bool  # #[doc(hidden)] (Rust) or a leading underscore (pyo3 export name)
    where: str  # "top" | "method" | "pyo3" | "python" | "export"
    owner: str | None = (
        None  # the Self type of an inherent-impl method (where "method") or the exposed pyclass of a #[pymethods] fn (where "pyo3")
    )


_PUB = re.compile(
    r"(?<![\w])pub(?!\s*\()\s+(?:async\s+|const\s+|unsafe\s+|extern\s+\"[^\"]*\"\s+)*(fn|struct|enum|trait|type)\s+([A-Za-z_]\w*)"
)
_IMPL = re.compile(r"\bimpl\b(?P<head>[^{;]*)\{")
_TYPE_DECL = re.compile(
    r"(?<![\w])(pub(?:\s*\([^)]*\))?\s+)?(struct|enum|trait|union|type)\s+([A-Za-z_]\w*)"
)


def _hidden(attrs: list[str]) -> bool:
    return any("doc(hidden)" in re.sub(r"\s+", "", a) for a in attrs)


def _inside(ranges: list[tuple[int, int]], pos: int) -> bool:
    return any(lo < pos < hi for lo, hi in ranges)


def _self_type(head: str) -> tuple[str | None, bool]:
    """(type name, is_trait_impl) of an `impl<...> [Trait for] Type<...>` head."""
    head = (
        re.sub(r"^\s*<.*?>\s*", "", head.strip(), count=1)
        if head.strip().startswith("<")
        else head.strip()
    )
    trait = re.search(r"\bfor\b", head) is not None
    if trait:
        head = head.split(" for ", 1)[-1]
    m = re.match(r"\s*(?:&\s*(?:'\w+\s*)?(?:mut\s+)?)?([A-Za-z_]\w*)", head)
    return (m.group(1) if m else None), trait


def _sectioned_enum_symbols(
    path: Path,
    code: str,
    skel: str,
    fn_bodies: list[tuple[int, int]],
    private_mods: list[tuple[int, int]],
) -> list[Symbol]:
    """Inventory the public enum produced by the one verified enum macro."""
    macro_path = (
        Path(__file__).resolve().parents[1] / "crates/antecedent-io-core/src/artifact_codec.rs"
    )
    macro_code, macro_skel = nontest_pair(macro_path)
    _, macro_items = rust_items(macro_path)
    definitions = [
        item
        for item in macro_items
        if item.kind == "macro" and item.name == "sectioned_artifact_enum"
    ]
    if len(definitions) != 1:
        return []
    definition = definitions[0]
    if [
        re.sub(r"\s+", "", attr) for attr in _attrs_before(macro_skel, macro_code, definition.start)
    ] != ["#[macro_export]"]:
        return []
    template = re.sub(r"\s+", "", macro_skel[definition.body[0] : definition.body[1]])
    if "#[non_exhaustive]pubenum$name{" not in template:
        return []
    _, items = rust_items(path.resolve())
    macro_ranges = [item.body for item in items if item.kind == "macro"]
    output: list[Symbol] = []
    for call in re.finditer(r"(?<![\w:])crate::sectioned_artifact_enum\s*!\s*([({\[])", skel):
        pos = call.start()
        if any(lo < pos < hi for lo, hi in (*fn_bodies, *private_mods, *macro_ranges)):
            continue
        if any(attr.startswith("#[cfg(") for attr in _attrs_before(skel, code, pos)):
            continue
        opening = call.end() - 1
        closing = _match_bracket(skel, opening)
        args = skel[opening + 1 : closing]
        match = re.match(r"\s*(?:#\s*\[[^\]]*\]\s*)*([A-Z][A-Za-z0-9_]*)\s*\{", args)
        if match:
            name = match[1]
            if not any(symbol.name == name and symbol.where == "top" for symbol in output):
                output.append(Symbol(name, "type", False, "top"))
    return output


def _counterfactual_index_newtypes(path: Path, code: str, skel: str):
    """Expand only the reviewed literal index-newtype template and invocations."""
    expected = REPO_ROOT / "crates/antecedent-identify/src/sid/counterfactual_transport.rs"
    if path.resolve() != expected.resolve():
        return None
    definitions = list(re.finditer(r"\bmacro_rules!\s+index_newtype\s*\{", skel))
    if len(definitions) != 1:
        return None
    definition = definitions[0]
    end = _match_bracket(skel, definition.end() - 1) + 1
    digest = hashlib.sha256(_compact_rust_tokens(code[definition.start():end]).encode()).hexdigest()
    if digest != "041d54ffb202ca1a66a9eb4bb72a6ef82a085dc183f0f6f2a93baa957c4fc6b4":
        return None
    calls = list(re.finditer(r"\bindex_newtype!\s*\{", skel))
    # Unknown delimiters, non-literal invocations or duplicate types invalidate
    # the whole proof; never silently hide an unrecognized macro definition.
    if len(re.findall(r"\bindex_newtype\s*!", skel)) != len(calls):
        return None
    names = []
    for call in calls:
        close = _match_bracket(skel, call.end() - 1)
        found = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*", skel[call.end():close])
        if found is None or found[1] in names:
            return None
        names.append(found[1])
    if not names:
        return None
    return (definition.start(), end), names


def rust_pub_items(path: Path, *, in_crate: bool = True) -> list[Symbol]:
    """Public non-test `pub fn/struct/enum/trait/type` items and re-exports:
    top-level, in public modules, and methods of inherent `impl` blocks of public
    types (each carries its owner type, so two types' same-named methods stay
    distinct). It also recognizes the enum emitted by the verified sectioned
    artifact macro. Items in fn bodies, private modules and non-public or trait
    impls are skipped."""
    code, skel = nontest_pair(path, in_crate=in_crate)
    if not code:
        return []
    _, items = rust_items(path.resolve())
    fn_bodies = [it.body for it in items if it.kind == "fn"]
    index_macro = _counterfactual_index_newtypes(path, code, skel)
    proven_macros = [index_macro[0]] if index_macro else []
    private_mods = [
        it.body
        for it in items
        if it.kind == "mod" and not re.search(r"\bpub\s+$", skel[max(0, it.start - 12) : it.start])
    ]
    types_public: dict[str, bool] = {}
    for m in _TYPE_DECL.finditer(skel):
        if not _inside(fn_bodies, m.start()):
            types_public[m.group(3)] = types_public.get(m.group(3), False) or (
                m.group(1) is not None and re.sub(r"\s+", "", m.group(1)) == "pub"
            )
    impls: list[tuple[int, int, str | None, bool]] = []
    for m in _IMPL.finditer(skel):
        name, trait = _self_type(m.group("head"))
        impls.append((m.end() - 1, _match_bracket(skel, m.end() - 1), name, trait))
    out: list[Symbol] = []
    seen: set[tuple[str, str | None]] = set()
    for m in _PUB.finditer(skel):
        pos = m.start()
        if _inside(fn_bodies, pos) or _inside(private_mods, pos) or _inside(proven_macros, pos):
            continue
        keyword_at = m.start(1)
        enclosing = [i for i in impls if i[0] < pos < i[1]]
        where = "top"
        owner: str | None = None
        if enclosing:
            lo, hi, tname, trait = max(enclosing, key=lambda i: i[0])
            if trait or m.group(1) != "fn" or not types_public.get(tname or "", True):
                continue
            if any(
                lo2 < pos < hi2 and (lo2, hi2) != (lo, hi) for lo2, hi2, *_ in enclosing if lo2 > lo
            ):
                continue
            where, owner = "method", tname
        name, item = m.group(2), m.group(1)
        if (name, owner) in seen:
            continue
        seen.add((name, owner))
        out.append(
            Symbol(
                name,
                "function"
                if item == "fn"
                else "type"
                if item in ("struct", "enum", "type")
                else "other",
                _hidden(_attrs_before(skel, code, keyword_at)),
                where,
                owner,
            )
        )
    if index_macro:
        for name in index_macro[1]:
            out.append(Symbol(name, "type", False, "top"))
            out.append(Symbol("new", "function", False, "method", name))
            out.append(Symbol("get", "function", False, "method", name))
    out.extend(_sectioned_enum_symbols(path, code, skel, fn_bodies, private_mods))
    # `pub use a::b::{X, y as z};` re-exports public API under the names it binds
    # (a glob `*` binds names this scan cannot see and is skipped).
    for m in _PUB_USE.finditer(skel):
        pos = m.start()
        if (
            _inside(fn_bodies, pos)
            or _inside(private_mods, pos)
            or any(lo < pos < hi for lo, hi, *_ in impls)
        ):
            continue
        hidden = _hidden(_attrs_before(skel, code, m.start(1)))
        for name in _use_names(m.group("tree")):
            if (name, None) in seen:
                continue
            seen.add((name, None))
            out.append(Symbol(name, "type" if name[:1].isupper() else "function", hidden, "top"))
    return out


_PUB_USE = re.compile(r"(?<![\w])pub(?!\s*\()\s+(use)\s+(?P<tree>[^;]*);")


def _use_names(tree: str) -> list[str]:
    """Names a `use` tree binds: the last path segment, or the `as` alias."""
    names = []
    for leaf in re.split(r"[{},]", tree):
        leaf = leaf.strip()
        if not leaf or leaf == "*" or leaf.endswith("::*"):
            continue
        alias = re.search(r"\bas\s+([A-Za-z_]\w*)\s*$", leaf)
        name = alias.group(1) if alias else leaf.rsplit("::", 1)[-1]
        if name in ("self", "super", "crate", "_") or not re.fullmatch(r"[A-Za-z_]\w*", name):
            continue
        if re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            continue  # a const/static: `pub const` items are not scanned either
        names.append(name)
    return names


_PY_FN = re.compile(
    r"#\[pyfunction(?P<args>[^\]]*)\](?P<attrs>(?:\s*#\[[^\]]*\])*)\s*(?:pub(?:\([^)]*\))?\s+)?fn\s+(?P<name>\w+)"
)
_PY_CLASS = re.compile(
    r"#\[pyclass(?P<args>[^\]]*)\](?P<attrs>(?:\s*#\[[^\]]*\])*)\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum)\s+(?P<name>\w+)"
)
_PY_METHODS = re.compile(r"#\[pymethods\]\s*impl\b(?P<head>[^{;]*)\{")
_PY_NAME = re.compile(r"""name\s*=\s*"([^"]+)\"""")


def pyo3_items(path: Path, *, in_crate: bool = True) -> list[Symbol]:
    """Names a pyo3 module exposes: `#[pyfunction]`s, `#[pyclass]`es and the
    methods of `#[pymethods]` impls (constructors and dunders are part of their
    class). A `#[pyo3(name = "x")]` renames; a leading underscore or
    `#[doc(hidden)]` marks the export internal. A method carries its class as
    `owner` (the exposed class name), so two classes' same-named getters stay
    distinct and a `surface_values` class can cover its own getters."""
    code, skel = nontest_pair(path, in_crate=in_crate)
    if not code:
        return []
    out: list[Symbol] = []
    seen: set[tuple[str, str | None]] = set()

    def add(name: str, kind: str, attrs: str, owner: str | None = None) -> None:
        renamed = _PY_NAME.search(attrs)
        exposed = renamed.group(1) if renamed else name
        if (exposed, owner) in seen:
            return
        seen.add((exposed, owner))
        hidden = exposed.startswith("_") or "doc(hidden)" in re.sub(r"\s+", "", attrs)
        out.append(Symbol(exposed, kind, hidden, "pyo3", owner))

    classes: dict[str, str] = {}  # rust struct name -> exposed class name
    for m in _PY_FN.finditer(code):
        # Every outer attribute of the fn: `#[pyfunction(name = "x")]` itself and a
        # `#[pyo3(name = "x")]` placed before or after it all rename the export.
        fn_at = skel.rfind("fn", m.end("attrs"), m.start("name"))
        attrs = (
            "".join(_attrs_before(skel, code, fn_at))
            if fn_at >= 0
            else m.group("args") + m.group("attrs")
        )
        add(m.group("name"), "function", attrs)
    for m in _PY_CLASS.finditer(code):
        add(m.group("name"), "type", m.group("args") + m.group("attrs"))
        renamed = _PY_NAME.search(m.group("args") + m.group("attrs"))
        classes[m.group("name")] = renamed.group(1) if renamed else m.group("name")
    for m in _PY_METHODS.finditer(skel):
        lo = m.end() - 1
        hi = _match_bracket(skel, lo)
        rust_name, _ = _self_type(m.group("head"))
        owner = classes.get(rust_name or "", rust_name)
        depth = 0
        for fm in re.finditer(r"[{}]|\bfn\s+(\w+)", skel[lo:hi]):
            if fm.group(0) == "{":
                depth += 1
            elif fm.group(0) == "}":
                depth -= 1
            elif depth == 1:
                name = fm.group(1)
                if name == "new" or (name.startswith("__") and name.endswith("__")):
                    continue
                attrs = "".join(_attrs_before(skel, code, lo + fm.start()))
                add(name, "function", attrs, owner)
    return out


def python_symbols(path: Path) -> dict[str, Symbol]:
    """Public top-level names of a Python file: `__all__` if it defines it (kinds
    resolved through `from .mod import Name`), else defs and classes without a
    leading underscore."""
    tree = ast.parse(path.read_text(errors="ignore", encoding="utf-8"))
    kinds = {
        node.name: "type" if isinstance(node, ast.ClassDef) else "function"
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            target = path.parent / (node.module.replace(".", "/") + ".py")
            if target.is_file():
                try:
                    inner = ast.parse(target.read_text(errors="ignore", encoding="utf-8"))
                except SyntaxError:
                    continue
                for sub in inner.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        for alias in node.names:
                            if alias.name == sub.name:
                                kinds.setdefault(
                                    alias.asname or alias.name,
                                    "type" if isinstance(sub, ast.ClassDef) else "function",
                                )
    for node in tree.body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets) and isinstance(
            node.value, (ast.List, ast.Tuple)
        ):
            return {
                e.value: Symbol(e.value, kinds.get(e.value, "other"), False, "python")
                for e in node.value.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            }
    return {
        name: Symbol(name, kind, False, "python")
        for name, kind in kinds.items()
        if not name.startswith("_")
    }


# -------------------------------------------------------------------- assertions

# An assertion macro or helper, a `panic!`, or an Err-forcing unwrap.
_RS_ASSERT = re.compile(
    r"\b(?:debug_|prop_)?assert\w*!|\bpanic!|\.expect_err\s*\(|\.unwrap_err\s*\(|\bassert_\w+\s*\("
)
_RS_TRIVIAL = re.compile(r"\b(?:debug_)?assert!\s*\(\s*(?:true|1\s*==\s*1)\s*[,)]")
_RS_ASSERT_ARGS = re.compile(r"\b(?:debug_|prop_)?assert\w*!\s*[(\[{]")


def rust_closure_parts(path: Path, name: str) -> tuple[str, str]:
    """(code, skeleton) of a Rust test's closure. A file outside the repo (a
    self-test's synthetic evidence) has no cargo target, so its closure is read
    from that one file."""
    path = path.resolve()
    try:
        return _rust_closure_parts(path, name)
    except ValueError:
        masked, items = rust_items(path)
        fns = {it.name: it for it in items if it.kind == "fn"}
        seen, todo, code, skel = set(), [name], [], []
        while todo:
            nm = todo.pop()
            if nm in seen or nm not in fns:
                continue
            seen.add(nm)
            lo, hi = fns[nm].body
            code.append(masked.code[lo:hi])
            skel.append(masked.skeleton[lo:hi])
            todo.extend(re.findall(r"(?<![.:\w])([A-Za-z_]\w*)\s*\(", masked.skeleton[lo:hi]))
        return "\n".join(code), "\n".join(skel)


def closure_code(path: Path, name: str) -> str:
    """Code of a cited test and the same-target helpers it reaches."""
    return closure(path, name) if path.suffix == ".py" else rust_closure_parts(path, name)[0]


def _rs_test_fns(path: Path, name: str):
    return [it for it in rust_items(path.resolve())[1] if it.kind == "fn" and it.name == name]


def rust_assertion_problems(path: Path, name: str) -> list[tuple[str, str]]:
    """(rule, message) for a cited Rust test that proves nothing: no assertion in
    its body or local helpers, or a `#[should_panic]` test (any panic passes it)."""
    fns = _rs_test_fns(path, name)
    if not fns:
        return []
    problems = []
    if any("should_panic" in re.sub(r"\s+", "", a) for a in fns[0].attrs):
        problems.append(
            (
                "evidence_should_panic",
                f"test {name} is #[should_panic]; any panic passes it, so it observes no value",
            )
        )
    _code, skel = rust_closure_parts(path, name)
    real = [m for m in _RS_ASSERT.finditer(skel) if not _RS_TRIVIAL.match(skel, m.start())]
    if not real:
        problems.append(
            (
                "evidence_no_assertion",
                f"test {name} contains no assertion (assert!/assert_eq!/assert_ne!/panic!/expect_err) in its body or helpers",
            )
        )
    return problems


def rust_assertion_args(path: Path, name: str) -> list[str]:
    """Argument text of every assertion macro in a Rust test's closure."""
    code, skel = rust_closure_parts(path, name)
    out = []
    for m in _RS_ASSERT_ARGS.finditer(skel):
        end = _match_bracket(skel, m.end() - 1)
        out.append(code[m.end() : end])
    return out


def _py_closure(path: Path, name: str) -> list[ast.AST]:
    tree = ast.parse(path.read_text(errors="ignore", encoding="utf-8"))
    return closure_nodes(tree, name)


_PY_CHECKS = {"raises", "warns", "fail", "deprecated_call"}


def _py_assertions(path: Path, name: str) -> list[ast.AST]:
    found: list[ast.AST] = []
    for fn in _py_closure(path, name):
        for node in ast.walk(fn):
            if isinstance(node, ast.Assert):
                if not (isinstance(node.test, ast.Constant) and node.test.value):
                    found.append(node)
            elif isinstance(node, ast.Call):
                callee = (
                    node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else getattr(node.func, "id", "")
                )
                if callee in _PY_CHECKS or callee.startswith("assert"):
                    found.append(node)
    return found


def python_assertion_problems(path: Path, name: str) -> list[tuple[str, str]]:
    try:
        if not _py_closure(path, name):
            return []
        if not _py_assertions(path, name):
            return [
                (
                    "evidence_no_assertion",
                    f"test {name} contains no assert statement or pytest.raises in its body or helpers",
                )
            ]
    except SyntaxError:
        return []
    return []


def assertion_problems(path: Path, name: str) -> list[tuple[str, str]]:
    if path.suffix == ".py":
        return python_assertion_problems(path, name)
    if path.suffix == ".rs":
        return rust_assertion_problems(path, name)
    return []


def assertion_texts(path: Path, name: str) -> list[str]:
    """Text of each assertion in a cited test (Rust macro arguments; Python assert
    statements and pytest.raises calls)."""
    if path.suffix == ".rs":
        return rust_assertion_args(path, name)
    try:
        return [ast.unparse(n) for n in _py_assertions(path, name)]
    except SyntaxError:
        return []


# ----------------------------------------------------------- charge in a loop

_LOOP = re.compile(
    r"\b(?:for|while|loop)\b|\.(?:try_for_each|for_each|try_fold|fold|map|filter_map|flat_map)\s*\("
)


_FN_QUALIFIER = re.compile(r"(?:async|const|unsafe|extern|\"C\")\Z")


def _fn_is_pub(skel: str, start: int) -> bool:
    """Whether the `fn` keyword at `start` carries a `pub` / `pub(...)` visibility."""
    k = start
    while True:
        while k > 0 and skel[k - 1].isspace():
            k -= 1
        if k == 0:
            return False
        if skel[k - 1] == ")":
            j, depth = k - 1, 0
            while j >= 0:
                depth += 1 if skel[j] == ")" else -1 if skel[j] == "(" else 0
                if depth == 0:
                    break
                j -= 1
            w = j
            while w > 0 and (skel[w - 1].isalnum() or skel[w - 1] == "_"):
                w -= 1
            return skel[w:j] == "pub"
        w = k
        while w > 0 and (skel[w - 1].isalnum() or skel[w - 1] in '_"'):
            w -= 1
        word = skel[w:k]
        if word == "pub":
            return True
        if _FN_QUALIFIER.match(word):
            k = w
            continue
        return False


def _trait_impl_ranges(skel: str) -> list[tuple[int, int]]:
    """Body spans of `impl Trait for Type` blocks: their fns are callable from
    outside the file without being `pub`."""
    out = []
    for m in _IMPL.finditer(skel):
        _, trait = _self_type(m.group("head"))
        if trait:
            out.append((m.end() - 1, _match_bracket(skel, m.end() - 1)))
    return out


def _referenced_elsewhere(name: str, path: Path) -> bool:
    """Whether another non-test file of `path`'s crate calls `name(`: a private fn
    reached from a child module (`super::name`) is live even without `pub`."""
    src = crate_src(path)
    if src is None:
        return False
    pattern = re.compile(rf"(?<![\w]){re.escape(name)}\s*(?:\(|::<)")
    for other in sorted(src.glob("**/*.rs")):
        if other.resolve() == path.resolve():
            continue
        if name not in other.read_text(errors="ignore", encoding="utf-8"):
            continue
        if pattern.search(nontest_pair(other, in_crate=True)[1]):
            return True
    return False


def charge_in_loop(path: Path, *, in_crate: bool = True) -> list[str]:
    """Names of non-test fns of `path` that call `.charge(` and either loop, call
    themselves, or are called by a fn that loops or recurses, and that live code
    can reach: the fn or one of its in-file callers is `pub`, a trait-impl
    method, or called by name from another file of the crate. A dead private fn
    that charges in a loop meters nothing."""
    code, skel = nontest_pair(path, in_crate=in_crate)
    if not code:
        return []
    fns = [it for it in rust_items(path.resolve())[1] if it.kind == "fn"]
    trait_impls = _trait_impl_ranges(skel)
    bodies: dict[str, list[str]] = {}
    roots: set[str] = set()
    for it in fns:
        body = skel[it.body[0] : it.body[1]]
        if not body.strip():
            continue
        bodies.setdefault(it.name, []).append(body)
        if _fn_is_pub(skel, it.start) or _inside(trait_impls, it.start):
            roots.add(it.name)
    callers: dict[str, set[str]] = {name: set() for name in bodies}
    for caller, bs in bodies.items():
        for callee in bodies:
            if callee != caller and any(re.search(rf"\b{re.escape(callee)}\s*\(", b) for b in bs):
                callers[callee].add(caller)

    def reachable(name: str) -> bool:
        seen, todo = set(), [name]
        while todo:
            nm = todo.pop()
            if nm in seen:
                continue
            seen.add(nm)
            if nm in roots or (in_crate and _referenced_elsewhere(nm, path)):
                roots.add(nm)
                return True
            todo.extend(callers[nm])
        return False

    looping = {
        name
        for name, bs in bodies.items()
        if any(
            _LOOP.search(b) or re.search(rf"(?<![.\w])(?:Self::|self\.)?{re.escape(name)}\s*\(", b)
            for b in bs
        )
    }
    out = []
    for name, bs in bodies.items():
        if not any(".charge(" in b for b in bs):
            continue
        called = any(caller in looping for caller in callers[name])
        if (name in looping or called) and reachable(name):
            out.append(name)
    return out
