"""Source analysis for scripts/check_promotion_records.py.

Pure functions over Rust and Python files: non-test source, string literals
(raw, byte and `\\`-continued Rust strings; Python f-strings, concatenations and
adjacent literals), public surfaces (Rust `pub` items and methods, pyo3 exports,
Python `__all__`), assertions in cited tests, and `.charge(` inside loops.
They report facts; the checker decides which fact is a violation.
"""

import ast
import re
from dataclasses import dataclass
from pathlib import Path

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

_crate_modules: dict[Path, dict[Path, bool]] = {}


def crate_src(path: Path) -> Path | None:
    for parent in path.parents:
        if parent.name == "src":
            return parent
    return None


def _compiled_test_only(path: Path) -> bool | None:
    """True if every target compiling `path` does so only under cfg(test); None if
    no target of its crate compiles it."""
    src = crate_src(path)
    if src is None:
        return None
    if src not in _crate_modules:
        merged: dict[Path, bool] = {}
        roots = [src / "lib.rs", src / "main.rs", *sorted((src / "bin").glob("*.rs"))]
        for top in roots:
            if top.is_file():
                for mod, test_only in target_modules(top.resolve()).items():
                    merged[mod] = merged.get(mod, True) and test_only
        _crate_modules[src] = merged
    return _crate_modules[src].get(path.resolve())


_nontest_cache: dict[Path, tuple[str, str]] = {}


def nontest_pair(path: Path, *, in_crate: bool = True) -> tuple[str, str]:
    """(code, skeleton) of `path` with comments removed and #[cfg(test)]
    modules/functions and #[test] functions blanked; both empty when the file is
    compiled only for tests or not compiled at all. `code` keeps string literals,
    `skeleton` blanks their contents; offsets agree."""
    key = path.resolve()
    if key in _nontest_cache:
        return _nontest_cache[key]
    out = ("", "")
    compiled = _compiled_test_only(path) is False if in_crate else True
    if compiled:
        masked, items = rust_items(key)
        if not re.search(r"#!\[cfg\(\s*test\s*\)\]", masked.code):
            code, skel = list(masked.code), list(masked.skeleton)
            for item in items:
                if item.kind not in ("fn", "mod"):
                    continue
                if _is_cfg_test(item.attrs) or any(re.sub(r"\s+", "", a) == "#[test]" for a in item.attrs):
                    for k in range(item.start, item.body[1]):
                        if code[k] != "\n":
                            code[k] = skel[k] = " "
            out = ("".join(code), "".join(skel))
    _nontest_cache[key] = out
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


def rust_literals(path: Path, *, in_crate: bool = True) -> list[Lit]:
    """Every string literal in non-test Rust source, with the const it initialises."""
    code, _ = nontest_pair(path, in_crate=in_crate)
    if not code:
        return []
    _, items = rust_items(path.resolve())
    consts = [it for it in items if it.kind == "const"]
    out: list[Lit] = []
    for at, value in _scan_literals(code):
        const = next((c.name for c in consts if c.body[0] <= at < c.body[1]), None)
        out.append(Lit(code.count("\n", 0, at) + 1, value, const))
    return out


def refusal_stage_literals(
    path: Path, *, in_crate: bool = True
) -> set[tuple[int, str]]:
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
            if prefix.count("{") == prefix.count("}") and prefix.count(
                "("
            ) == prefix.count(")"):
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


def finite_refusal_helpers(
    path: Path, *, in_crate: bool = True
) -> tuple[list[Lit], set[int]]:
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
        body = (
            code[item.body[0] : item.body[1]]
            .strip()
            .removeprefix("{")
            .removesuffix("}")
        )
        matched = body_pattern.fullmatch(body)
        if not matched or (
            (matched["constructor"] == "refused_with_fields") != bool(matched["fields"])
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
                if lit.value == matched["namespace"] + ".{detail}: {message}"
                and code.count("\n", 0, item.body[0]) + 1
                <= lit.line
                <= code.count("\n", 0, item.body[1]) + 1
                and sum(
                    other.line == lit.line and other.value == lit.value
                    for other in literals
                )
                == 1
            )
    return output, proven_lines


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
            problems.append(
                "shared namespace needs exactly namespace, owner_record, sources"
            )
            continue
        ns, owner, sources = (
            declaration[k] for k in ("namespace", "owner_record", "sources")
        )
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
            not p.endswith(".rs") or "/src/" not in p or ".." in Path(p).parts
            for p in sources
        ):
            problems.append(
                f"{ns}: sources must exactly equal emitting production Rust files"
            )
        owners = [r for r in records if r.get("id") == owner]
        if len(owners) != 1 or owners[0].get("status") not in {
            "promoted",
            "in_progress",
        }:
            problems.append(f"{ns}: missing implemented namespace owner")
            continue
        if declaration not in owners[0].get("shared_refusal_namespaces", []):
            problems.append(
                f"{ns}: owner must declare identical sources and self ownership"
            )
        owned = {r.get("detail"): r.get("code") for r in owners[0].get("refusals", [])}
        if (
            any(
                owned.get(r.get("detail")) != r.get("code")
                for r in record.get("refusals", [])
                if r.get("detail", "").startswith(ns + ".")
            )
            or not set(emissions.get(ns, {})) <= owned.keys()
        ):
            problems.append(
                f"{ns}: every emitted refusal pair must be declared by owner"
            )
    if len(namespaces - seen) > 1:
        problems.append("refusal details must share one primary namespace")
    return problems


def defensive_carrier_route_problems(root: Path) -> list[str]:
    """Factory and fresh consumer permit exactly the six source-report routes."""
    native, native_skel = nontest_pair(root / "python/src/measured_inference_api.rs")
    io, _ = nontest_pair(root / "crates/antecedent-io/src/measured_inference.rs")
    expected = {
        "joint_bayesian",
        "learned_joint",
        "nested_fisher",
        "nested_bayesian",
        "sampled_recovery",
        "temporal_interval",
    }
    _, native_items = rust_items(root / "python/src/measured_inference_api.rs")
    _, io_items = rust_items(root / "crates/antecedent-io/src/measured_inference.rs")
    source = [
        native[i.body[0] : i.body[1]]
        for i in native_items
        if i.kind == "fn" and i.name == "source_report"
    ]
    consume = [
        io[i.body[0] : i.body[1]]
        for i in io_items
        if i.kind == "fn" and i.name == "consume"
    ]
    arms = lambda text: set(re.findall(r'"([a-z_]+)"\s*=>', text))
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
            "defensive source-only claim requires the exact six factory/fresh-consumer route match and no public constructor/setter"
        ]
    # Every native IO factory call uses a literal member of the same finite
    # route set. Unknown/dynamic factory arguments and escaped aliases reject
    # the defensive-only claim even when the consumer still rejects them.
    factory_routes = set()
    constructor_count = 0
    for path in (root / "crates/antecedent-io/src").glob("**/*.rs"):
        code, skeleton = nontest_pair(path)
        constructor_count += len(
            re.findall(r"\bMeasuredInference\s*\{", skeleton)
        ) - len(re.findall(r"\b(?:struct|impl)\s+MeasuredInference\s*\{", skeleton))
        if re.search(r"\bauthorize\s+as\b", skeleton):
            return [
                "defensive factory proof cannot permit an escaped authority helper alias"
            ]
        code = re.sub(r"\buse\b[^;]+;", lambda m: " " * len(m[0]), code)
        for call in re.finditer(r"\bauthorize\b", code):
            if re.search(r"\bfn\s*$", code[: call.start()]):
                continue
            argument = re.match(r'\s*\(\s*"([a-z_]+)"\s*,', code[call.end() :])
            if not argument or argument[1] not in expected:
                return [
                    "defensive factory proof requires a finite literal route at every native authority call"
                ]
            factory_routes.add(argument[1])
    if factory_routes != expected or constructor_count != 1:
        return [
            "defensive factory proof requires exactly the original six routes and one private authority constructor"
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
        or re.search(
            r"#\[derive\([^]]*Deserialize[^]]*\)\]\s*pub struct MeasuredInference\b", io
        )
    ):
        return [
            "defensive carrier route must remain opaque, immutable and non-deserializable"
        ]
    return []


def input_exception_evidence_problems(
    entry: dict, path: Path, assertion: str
) -> list[str]:
    """Require class assertion plus absent reason code, including finite pytest cases."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == assertion
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
    assertions = "\n".join(
        ast.unparse(n) for n in ast.walk(fn) if isinstance(n, ast.Assert)
    )
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
            n.args and ast.unparse(n.args[0]) == "getattr(errors, exception)"
            for n in raises
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
    return [
        "class-only runtime evidence must exercise the exact declared class/detail case"
    ]


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
        return [
            "live precondition/resource branch cannot be declared defensive_source_only"
        ]
    if detail == "measured_inference.source_report_not_supported":
        if defensive is not True:
            return [
                "unreachable source-report guard must be explicitly defensive_source_only"
            ]
        problems = defensive_carrier_route_problems(root)
        if problems:
            return problems
    code, skel = nontest_pair(root / source)
    helper = re.escape(str(entry["helper"]))
    literal = re.escape(detail)
    call = re.compile(
        helper + r'\s*\(\s*(?:format!\(\s*)?"' + literal + r'(?::[^"\n]*)?"'
    )
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
        return [
            "this class-only detail is also emitted outside its proven typed helper"
        ]
    for wrapper in re.finditer(r"\bwith_reason_code\s*\(", skel):
        close = _match_bracket(skel, wrapper.end() - 1)
        if any(wrapper.end() <= match.start() < close for match in matches):
            return ["class-only constructor is wrapped in a reason-code assignment"]
    # The registered helper may change; prove the actual current mapping rather
    # than equating arbitrary functions with ValueError by their spelling.
    if pair[0] == "CausalValueError":
        lib, _ = nontest_pair(root / "python/src/lib.rs")
        _, items = rust_items(root / "python/src/lib.rs")
        bodies = [
            lib[i.body[0] : i.body[1]]
            for i in items
            if i.kind == "fn" and i.name == "value_err"
        ]
        if (
            len(bodies) != 1
            or "VALUE_ERROR_CLASS" not in bodies[0]
            or "CausalValidateError::new_err" not in bodies[0]
            or "reason_code" in bodies[0]
        ):
            return [
                "value_err no longer proves registered value-error mapping without a reason code"
            ]
    if pair[0] == "CausalValueError":
        errors = ast.parse(
            (root / "python/antecedent/errors.py").read_text(encoding="utf-8")
        )
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
            and [ast.unparse(base) for base in n.bases]
            == ["CausalValidateError", "ValueError"]
        ]
        if len(registrations) != 1 or len(classes) != 1:
            return [
                "registered value-error class mapping no longer matches the native helper"
            ]
    return []


def ignored_test_literals(path: Path) -> dict[str, list[str]]:
    """Test name -> string literals in the body of every `#[ignore]`d fn of a Rust
    test file, comments excluded (the coverage-record ids a calibration test emits)."""
    masked, items = rust_items(path.resolve())
    out: dict[str, list[str]] = {}
    for it in items:
        if it.kind != "fn" or not any(re.sub(r"\s+", "", a).startswith("#[ignore") for a in it.attrs):
            continue
        lo, hi = it.body
        out.setdefault(it.name, []).extend(value for _, value in _scan_literals(masked.code, lo, hi))
    return out


def python_fn_literals(path: Path) -> dict[str, list[str]]:
    """Function name -> string constants in its body, for every def of a Python file."""
    try:
        tree = ast.parse(path.read_text(errors="ignore"))
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
            v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else "{}" for v in node.values
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
        tree = ast.parse(path.read_text(errors="ignore"))
    except SyntaxError:
        return []
    # Documentation is not an emitted runtime string. In particular, a sentence
    # ending in a namespace word such as "scenarios." is not a dynamic refusal.
    # Keep ordinary constants and actual f-strings/concatenations fully scanned.
    docstrings: set[int] = set()
    for scope in ast.walk(tree):
        if isinstance(scope, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = scope.body[0] if scope.body else None
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                docstrings.add(id(first.value))
    consts: dict[int, str] = {}
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        value = getattr(node, "value", None)
        if value is not None and any(isinstance(t, ast.Name) for t in targets):
            name = next(t.id for t in targets if isinstance(t, ast.Name))
            for sub in ast.walk(value):
                consts.setdefault(id(sub), name)
    seen: set[tuple[int, str]] = set()
    out: list[Lit] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Constant, ast.JoinedStr, ast.BinOp)) and id(node) not in docstrings:
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
    owner: str | None = None  # the Self type of an inherent-impl method (where "method") or the exposed pyclass of a #[pymethods] fn (where "pyo3")


_PUB = re.compile(
    r"(?<![\w])pub(?!\s*\()\s+(?:async\s+|const\s+|unsafe\s+|extern\s+\"[^\"]*\"\s+)*(fn|struct|enum|trait)\s+([A-Za-z_]\w*)"
)
_IMPL = re.compile(r"\bimpl\b(?P<head>[^{;]*)\{")
_TYPE_DECL = re.compile(r"(?<![\w])(pub(?:\s*\([^)]*\))?\s+)?(struct|enum|trait|union|type)\s+([A-Za-z_]\w*)")


def _hidden(attrs: list[str]) -> bool:
    return any("doc(hidden)" in re.sub(r"\s+", "", a) for a in attrs)


def _inside(ranges: list[tuple[int, int]], pos: int) -> bool:
    return any(lo < pos < hi for lo, hi in ranges)


def _self_type(head: str) -> tuple[str | None, bool]:
    """(type name, is_trait_impl) of an `impl<...> [Trait for] Type<...>` head."""
    head = re.sub(r"^\s*<.*?>\s*", "", head.strip(), count=1) if head.strip().startswith("<") else head.strip()
    trait = re.search(r"\bfor\b", head) is not None
    if trait:
        head = head.split(" for ", 1)[-1]
    m = re.match(r"\s*(?:&\s*(?:'\w+\s*)?(?:mut\s+)?)?([A-Za-z_]\w*)", head)
    return (m.group(1) if m else None), trait


def rust_pub_items(path: Path, *, in_crate: bool = True) -> list[Symbol]:
    """Public non-test `pub fn/struct/enum/trait` items and `pub use` re-exports:
    top-level, in public modules, and methods of inherent `impl` blocks of public
    types (each carries its owner type, so two types' same-named methods stay
    distinct). Items in fn bodies, in private modules and in impls of non-public
    or trait types are not public API and are skipped."""
    code, skel = nontest_pair(path, in_crate=in_crate)
    if not code:
        return []
    _, items = rust_items(path.resolve())
    fn_bodies = [it.body for it in items if it.kind == "fn"]
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
        if _inside(fn_bodies, pos) or _inside(private_mods, pos):
            continue
        keyword_at = m.start(1)
        enclosing = [i for i in impls if i[0] < pos < i[1]]
        where = "top"
        owner: str | None = None
        if enclosing:
            lo, hi, tname, trait = max(enclosing, key=lambda i: i[0])
            if trait or m.group(1) != "fn" or not types_public.get(tname or "", True):
                continue
            if any(lo2 < pos < hi2 and (lo2, hi2) != (lo, hi) for lo2, hi2, *_ in enclosing if lo2 > lo):
                continue
            where, owner = "method", tname
        name, item = m.group(2), m.group(1)
        if (name, owner) in seen:
            continue
        seen.add((name, owner))
        out.append(
            Symbol(
                name,
                "function" if item == "fn" else "type" if item in ("struct", "enum") else "other",
                _hidden(_attrs_before(skel, code, keyword_at)),
                where,
                owner,
            )
        )
    # `pub use a::b::{X, y as z};` re-exports public API under the names it binds
    # (a glob `*` binds names this scan cannot see and is skipped).
    for m in _PUB_USE.finditer(skel):
        pos = m.start()
        if _inside(fn_bodies, pos) or _inside(private_mods, pos) or any(lo < pos < hi for lo, hi, *_ in impls):
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
        attrs = "".join(_attrs_before(skel, code, fn_at)) if fn_at >= 0 else m.group("args") + m.group("attrs")
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
    tree = ast.parse(path.read_text(errors="ignore"))
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
                    inner = ast.parse(target.read_text(errors="ignore"))
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
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets) and isinstance(
            node.value, (ast.List, ast.Tuple)
        ):
            return {
                e.value: Symbol(e.value, kinds.get(e.value, "other"), False, "python")
                for e in node.value.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            }
    return {
        name: Symbol(name, kind, False, "python") for name, kind in kinds.items() if not name.startswith("_")
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
            ("evidence_should_panic", f"test {name} is #[should_panic]; any panic passes it, so it observes no value")
        )
    _code, skel = rust_closure_parts(path, name)
    real = [m for m in _RS_ASSERT.finditer(skel) if not _RS_TRIVIAL.match(skel, m.start())]
    if not real:
        problems.append(
            ("evidence_no_assertion", f"test {name} contains no assertion (assert!/assert_eq!/assert_ne!/panic!/expect_err) in its body or helpers")
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
    tree = ast.parse(path.read_text(errors="ignore"))
    defs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    seen: set[str] = set()
    todo, out = [name], []
    while todo:
        nm = todo.pop()
        if nm in seen or nm not in defs:
            continue
        seen.add(nm)
        out.append(defs[nm])
        todo.extend(n.id for n in ast.walk(defs[nm]) if isinstance(n, ast.Name))
    return out


_PY_CHECKS = {"raises", "warns", "fail", "deprecated_call"}


def _py_assertions(path: Path, name: str) -> list[ast.AST]:
    found: list[ast.AST] = []
    for fn in _py_closure(path, name):
        for node in ast.walk(fn):
            if isinstance(node, ast.Assert):
                if not (isinstance(node.test, ast.Constant) and node.test.value):
                    found.append(node)
            elif isinstance(node, ast.Call):
                callee = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
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

_LOOP = re.compile(r"\b(?:for|while|loop)\b|\.(?:try_for_each|for_each|try_fold|fold|map|filter_map|flat_map)\s*\(")


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
        if name not in other.read_text(errors="ignore"):
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
        if any(_LOOP.search(b) or re.search(rf"(?<![.\w])(?:Self::|self\.)?{re.escape(name)}\s*\(", b) for b in bs)
    }
    out = []
    for name, bs in bodies.items():
        if not any(".charge(" in b for b in bs):
            continue
        called = any(caller in looping for caller in callers[name])
        if (name in looping or called) and reachable(name):
            out.append(name)
    return out
