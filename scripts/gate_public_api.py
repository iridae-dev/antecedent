#!/usr/bin/env python3
"""Public-API regression gate.

Builds a deterministic listing of every published workspace crate's public
items from rustdoc JSON (default features, the repo-pinned toolchain with
``RUSTC_BOOTSTRAP=1``), compares it with the committed snapshot under
``parity/public_api/<crate>.txt`` and fails with a readable diff when the
surface changed. Regenerate deliberately with ``--write``; the snapshot diff is
the reviewed record of the API change.

Usage:
  gate_public_api.py [--write] [--crate NAME ...] [--json-dir DIR]
"""

from __future__ import annotations

import argparse
import difflib
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "parity" / "public_api"
FORMAT_VERSION = 37  # rustdoc JSON format of the pinned 1.85 toolchain

HEADER = (
    "# Public API snapshot (rustdoc JSON, default features). Regenerate with\n"
    "#   python3 scripts/gate_public_api.py --write\n"
    "# One line per public item; do not edit by hand.\n"
)


def published_crates() -> list[str]:
    ws = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    names: list[str] = []
    for member in ws["workspace"]["members"]:
        manifest = ROOT / member / "Cargo.toml"
        pkg = tomllib.loads(manifest.read_text(encoding="utf-8"))["package"]
        if pkg.get("publish", True) is False:
            continue
        if not (ROOT / member / "src" / "lib.rs").exists():
            continue
        names.append(pkg["name"])
    return sorted(names)


def run_rustdoc(crates: list[str], target_dir: Path | None) -> Path:
    env = dict(os.environ)
    env["RUSTC_BOOTSTRAP"] = "1"
    env["RUSTDOCFLAGS"] = (
        "-Z unstable-options --output-format json --document-hidden-items -A warnings"
    )
    if target_dir is not None:
        env["CARGO_TARGET_DIR"] = str(target_dir)
    cmd = ["cargo", "doc", "--no-deps", "--lib", "--quiet"]
    for name in crates:
        cmd += ["-p", name]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    out = target_dir or Path(env.get("CARGO_TARGET_DIR", ROOT / "target"))
    return out / "doc"


# ---------------------------------------------------------------------------
# rustdoc JSON rendering


class Lister:
    def __init__(self, doc: dict[str, Any]) -> None:
        if doc["format_version"] != FORMAT_VERSION:
            raise SystemExit(
                f"rustdoc JSON format {doc['format_version']} != {FORMAT_VERSION}; "
                "update scripts/gate_public_api.py for the new toolchain"
            )
        self.doc = doc
        self.index: dict[str, Any] = doc["index"]
        self.paths: dict[str, Any] = doc["paths"]
        self.canon: dict[str, str] = {}  # item id -> canonical public path
        self.lines: set[str] = set()

    # -- public path discovery -------------------------------------------------

    def walk(self) -> None:
        root = self.index[str(self.doc["root"])]
        found: dict[str, list[str]] = {}
        self._module(root, [root["name"]], found, set())
        for item_id, paths in found.items():
            self.canon[item_id] = sorted(paths)[0]
        self.found = found

    def _module(self, module, path, found, seen_mods) -> None:
        mid = str(module["id"])
        if mid in seen_mods:
            return
        seen_mods = seen_mods | {mid}
        for child_id in module["inner"]["module"]["items"]:
            child = self.index.get(str(child_id))
            if child is None:
                continue
            self._item(child, path, found, seen_mods)

    def _item(self, item, path, found, seen_mods) -> None:
        kind = next(iter(item["inner"]))
        if kind == "use":
            use = item["inner"]["use"]
            target = self.index.get(str(use["id"])) if use["id"] is not None else None
            if use["is_glob"]:
                if target is not None and next(iter(target["inner"])) == "module":
                    self._module(target, path, found, seen_mods)
                elif target is not None:
                    self._enum_glob(target, path, found)
                return
            name = use["name"]
            if target is None:
                found.setdefault(f"ext:{use['source']}", []).append("::".join([*path, name]))
                self.lines.add(f"pub use {'::'.join([*path, name])} = {use['source']}")
                return
            self._item_named(target, name, path, found, seen_mods)
            return
        if item.get("name") is None:
            return
        self._item_named(item, item["name"], path, found, seen_mods)

    def _enum_glob(self, target, path, found) -> None:
        # `pub use Enum::*`: variants become items of the enclosing module.
        if next(iter(target["inner"])) != "enum":
            return
        for vid in target["inner"]["enum"]["variants"]:
            variant = self.index.get(str(vid))
            if variant is not None:
                found.setdefault(str(vid), []).append("::".join([*path, variant["name"]]))

    def _item_named(self, item, name, path, found, seen_mods) -> None:
        kind = next(iter(item["inner"]))
        full = [*path, name]
        if kind == "module":
            found.setdefault(str(item["id"]), []).append("::".join(full))
            self._module(item, full, found, seen_mods)
            return
        found.setdefault(str(item["id"]), []).append("::".join(full))

    # -- type rendering --------------------------------------------------------

    def path_name(self, item_id: int | str, fallback: str) -> str:
        sid = str(item_id)
        if sid in self.canon:
            return self.canon[sid]
        info = self.paths.get(sid)
        if info is not None:
            return "::".join(info["path"])
        return fallback

    def rpath(self, p: dict[str, Any]) -> str:
        name = self.path_name(p["id"], p["name"])
        return name + self.rargs(p.get("args"))

    def rargs(self, args) -> str:
        if not args:
            return ""
        if "angle_bracketed" in args:
            ab = args["angle_bracketed"]
            parts = [self.rgarg(a) for a in ab["args"]]
            for c in ab["constraints"]:
                parts.append(self.rconstraint(c))
            return f"<{', '.join(parts)}>" if parts else ""
        pr = args["parenthesized"]
        ins = ", ".join(self.rtype(t) for t in pr["inputs"])
        out = f" -> {self.rtype(pr['output'])}" if pr.get("output") else ""
        return f"({ins}){out}"

    def rgarg(self, a) -> str:
        if "lifetime" in a:
            return a["lifetime"]
        if "type" in a:
            return self.rtype(a["type"])
        if "const" in a:
            return self.rconst(a["const"])
        return "_"

    def rconst(self, c) -> str:
        return c.get("value") or c.get("expr") or "_"

    def rconstraint(self, c) -> str:
        head = c["name"] + self.rargs(c.get("args"))
        b = c["binding"]
        if "equality" in b:
            eq = b["equality"]
            return (
                f"{head} = {self.rtype(eq['type']) if 'type' in eq else self.rconst(eq['const'])}"
            )
        return f"{head}: {self.rbounds(b['constraint'])}"

    def rbounds(self, bounds) -> str:
        return " + ".join(self.rbound(b) for b in bounds)

    def rbound(self, b) -> str:
        if "trait_bound" in b:
            tb = b["trait_bound"]
            hr = ""
            if tb["generic_params"]:
                hr = f"for<{', '.join(self.rgparam_head(g) for g in tb['generic_params'])}> "
            mod = {"none": "", "maybe": "?", "maybe_const": "~const "}[tb["modifier"]]
            return f"{hr}{mod}{self.rpath(tb['trait'])}"
        if "outlives" in b:
            return b["outlives"]
        return "use<..>"

    def rgparam_head(self, g) -> str:
        return g["name"]

    def rtype(self, t) -> str:
        k, v = next(iter(t.items()))
        if k == "resolved_path":
            return self.rpath(v)
        if k == "generic" or k == "primitive":
            return v
        if k == "borrowed_ref":
            lt = f"{v['lifetime']} " if v.get("lifetime") else ""
            mut = "mut " if v["is_mutable"] else ""
            return f"&{lt}{mut}{self.rtype(v['type'])}"
        if k == "tuple":
            inner = ", ".join(self.rtype(x) for x in v)
            return f"({inner},)" if len(v) == 1 else f"({inner})"
        if k == "slice":
            return f"[{self.rtype(v)}]"
        if k == "array":
            return f"[{self.rtype(v['type'])}; {v['len']}]"
        if k == "raw_pointer":
            return f"*{'mut' if v['is_mutable'] else 'const'} {self.rtype(v['type'])}"
        if k == "impl_trait":
            return f"impl {self.rbounds(v)}"
        if k == "dyn_trait":
            parts = [self.rpath(x["trait"]) for x in v["traits"]]
            if v.get("lifetime"):
                parts.append(v["lifetime"])
            return f"dyn {' + '.join(parts)}"
        if k == "function_pointer":
            sig = v["sig"]
            ins = ", ".join(self.rtype(ty) for _, ty in sig["inputs"])
            out = f" -> {self.rtype(sig['output'])}" if sig.get("output") else ""
            hdr = v["header"]
            pre = "unsafe " if hdr["is_unsafe"] else ""
            return f"{pre}fn({ins}){out}"
        if k == "qualified_path":
            tr = f" as {self.rpath(v['trait'])}" if v.get("trait") else ""
            return f"<{self.rtype(v['self_type'])}{tr}>::{v['name']}{self.rargs(v.get('args'))}"
        if k == "pat":
            return self.rtype(v["type"])
        return "_"

    # -- generics --------------------------------------------------------------

    def rgenerics(self, g) -> tuple[str, str]:
        params = []
        for p in g["params"]:
            kind = p["kind"]
            if "lifetime" in kind:
                outl = kind["lifetime"]["outlives"]
                params.append(p["name"] + (f": {' + '.join(outl)}" if outl else ""))
            elif "type" in kind:
                if kind["type"].get("is_synthetic"):
                    continue
                s = p["name"]
                if kind["type"]["bounds"]:
                    s += f": {self.rbounds(kind['type']['bounds'])}"
                if kind["type"].get("default"):
                    s += f" = {self.rtype(kind['type']['default'])}"
                params.append(s)
            else:
                c = kind["const"]
                s = f"const {p['name']}: {self.rtype(c['type'])}"
                if c.get("default"):
                    s += f" = {c['default']}"
                params.append(s)
        where = []
        for w in g["where_predicates"]:
            if "bound_predicate" in w:
                bp = w["bound_predicate"]
                where.append(f"{self.rtype(bp['type'])}: {self.rbounds(bp['bounds'])}")
            elif "lifetime_predicate" in w:
                lp = w["lifetime_predicate"]
                where.append(f"{lp['lifetime']}: {' + '.join(lp['outlives'])}")
            else:
                eq = w["eq_predicate"]
                where.append(f"{self.rtype(eq['lhs'])} == {self.rtype(eq['rhs'])}")
        head = f"<{', '.join(params)}>" if params else ""
        tail = f" where {', '.join(sorted(where))}" if where else ""
        return head, tail

    def rfn(self, item, name: str) -> str:
        f = item["inner"]["function"]
        head, tail = self.rgenerics(f["generics"])
        hdr = f["header"]
        quals = ""
        if hdr["is_const"]:
            quals += "const "
        if hdr["is_async"]:
            quals += "async "
        if hdr["is_unsafe"]:
            quals += "unsafe "
        ins = []
        for pname, ty in f["sig"]["inputs"]:
            rendered = self.rtype(ty)
            if pname == "self":
                if rendered == "Self":
                    ins.append("self")
                elif rendered == "&Self":
                    ins.append("&self")
                elif rendered == "&mut Self":
                    ins.append("&mut self")
                else:
                    ins.append(f"self: {rendered}")
            else:
                ins.append(f"{pname}: {rendered}")
        out = f" -> {self.rtype(f['sig']['output'])}" if f["sig"].get("output") else ""
        return f"{quals}fn {name}{head}({', '.join(ins)}){out}{tail}"

    # -- item listing ----------------------------------------------------------

    def attrs(self, item) -> str:
        pre = ""
        for a in item.get("attrs", []):
            if "doc(hidden)" in a:
                pre += "#[doc(hidden)] "
            if "non_exhaustive" in a:
                pre += "#[non_exhaustive] "
        return pre

    def emit(self, line: str) -> None:
        self.lines.add(line)

    def list_all(self) -> list[str]:
        self.walk()
        for item_id, paths in sorted(self.found.items()):
            if item_id.startswith("ext:"):
                continue
            item = self.index[item_id]
            canon = self.canon[item_id]
            for alias in paths:
                if alias != canon:
                    self.emit(f"pub use {alias} = {canon}")
            self.list_item(item, canon)
        self.list_impls()
        return sorted(self.lines)

    def list_item(self, item, path: str) -> None:
        kind = next(iter(item["inner"]))
        a = self.attrs(item)
        v = item["inner"][kind]
        if kind == "module":
            self.emit(f"{a}pub mod {path}")
        elif kind == "function":
            self.emit(f"{a}pub {self.rfn(item, path)}")
        elif kind == "constant":
            self.emit(f"{a}pub const {path}: {self.rtype(v['type'])}")
        elif kind == "static":
            m = "mut " if v["is_mutable"] else ""
            self.emit(f"{a}pub static {m}{path}: {self.rtype(v['type'])}")
        elif kind == "type_alias":
            head, tail = self.rgenerics(v["generics"])
            self.emit(f"{a}pub type {path}{head} = {self.rtype(v['type'])}{tail}")
        elif kind == "macro":
            self.emit(f"{a}macro {path}!")
        elif kind == "proc_macro":
            self.emit(f"{a}proc_macro {path}")
        elif kind in ("struct", "union"):
            self.list_struct(item, path, a, kind)
        elif kind == "enum":
            self.list_enum(item, path, a)
        elif kind == "variant":
            self.emit(f"{a}pub variant {path}")
        elif kind == "trait":
            self.list_trait(item, path, a)
        else:
            self.emit(f"{a}pub {kind} {path}")

    def list_struct(self, item, path, a, kind) -> None:
        s = item["inner"][kind]
        head, tail = self.rgenerics(s["generics"])
        self.emit(f"{a}pub {kind} {path}{head}{tail}")
        sk = s["kind"] if kind == "struct" else None
        if kind == "union":
            fields, stripped = s["fields"], s.get("has_stripped_fields", False)
            tuple_like = False
        elif sk == "unit":
            return
        elif "tuple" in sk:
            fields, stripped, tuple_like = sk["tuple"], False, True
        else:
            fields = sk["plain"]["fields"]
            stripped = sk["plain"]["has_stripped_fields"]
            tuple_like = False
        for i, fid in enumerate(fields):
            if fid is None:
                stripped = True
                continue
            f = self.index.get(str(fid))
            if f is None:
                stripped = True
                continue
            label = str(i) if tuple_like else f["name"]
            self.emit(f"pub field {path}::{label}: {self.rtype(f['inner']['struct_field'])}")
        if stripped:
            self.emit(f"pub struct {path} has private fields")

    def list_enum(self, item, path, a) -> None:
        e = item["inner"]["enum"]
        head, tail = self.rgenerics(e["generics"])
        self.emit(f"{a}pub enum {path}{head}{tail}")
        for vid in e["variants"]:
            var = self.index[str(vid)]
            va = self.attrs(var)
            vk = var["inner"]["variant"]["kind"]
            vpath = f"{path}::{var['name']}"
            if vk == "plain":
                self.emit(f"{va}pub variant {vpath}")
            elif "tuple" in vk:
                tys = []
                for fid in vk["tuple"]:
                    f = self.index.get(str(fid)) if fid is not None else None
                    tys.append(self.rtype(f["inner"]["struct_field"]) if f else "_")
                self.emit(f"{va}pub variant {vpath}({', '.join(tys)})")
            else:
                fs = []
                for fid in vk["struct"]["fields"]:
                    f = self.index[str(fid)]
                    fs.append(f"{f['name']}: {self.rtype(f['inner']['struct_field'])}")
                self.emit(f"{va}pub variant {vpath} {{ {', '.join(fs)} }}")

    def list_trait(self, item, path, a) -> None:
        t = item["inner"]["trait"]
        head, tail = self.rgenerics(t["generics"])
        sup = f": {self.rbounds(t['bounds'])}" if t["bounds"] else ""
        quals = ("unsafe " if t["is_unsafe"] else "") + ("auto " if t["is_auto"] else "")
        self.emit(f"{a}pub {quals}trait {path}{head}{sup}{tail}")
        for mid in t["items"]:
            m = self.index.get(str(mid))
            if m is None:
                continue
            mk = next(iter(m["inner"]))
            mpath = f"{path}::{m['name']}"
            if mk == "function":
                dflt = " {default}" if m["inner"]["function"]["has_body"] else ""
                self.emit(f"pub trait fn {self.rfn(m, mpath)}{dflt}")
            elif mk == "assoc_type":
                at = m["inner"]["assoc_type"]
                bounds = f": {self.rbounds(at['bounds'])}" if at["bounds"] else ""
                dflt = f" = {self.rtype(at['type'])}" if at.get("type") else ""
                self.emit(f"pub trait type {mpath}{bounds}{dflt}")
            elif mk == "assoc_const":
                ac = m["inner"]["assoc_const"]
                self.emit(f"pub trait const {mpath}: {self.rtype(ac['type'])}")

    def list_impls(self) -> None:
        for item in self.index.values():
            if item["crate_id"] != 0 or next(iter(item["inner"])) != "impl":
                continue
            im = item["inner"]["impl"]
            if im.get("blanket_impl") is not None:
                continue
            for_ty = im["for"]
            if "resolved_path" in for_ty:
                tid = str(for_ty["resolved_path"]["id"])
                own_local = tid in self.canon
            else:
                own_local = False
            trait = im.get("trait")
            if trait is not None:
                tid = str(trait["id"])
                local_trait = tid in self.canon
                if not (own_local or local_trait):
                    continue
            elif not own_local:
                continue
            # impls of crate-private types (stripped) are not public API.
            head, tail = self.rgenerics(im["generics"])
            neg = "!" if im["is_negative"] else ""
            if trait is None:
                ty = self.rtype(for_ty)
                for mid in im["items"]:
                    m = self.index.get(str(mid))
                    if m is None or m.get("visibility") != "public":
                        continue
                    mk = next(iter(m["inner"]))
                    a = self.attrs(m)
                    if mk == "function":
                        self.emit(f"{a}pub fn <{ty}>::{self.rfn(m, m['name'])}")
                    elif mk == "constant":
                        self.emit(
                            f"{a}pub const <{ty}>::{m['name']}: "
                            f"{self.rtype(m['inner']['constant']['type'])}"
                        )
                    elif mk == "assoc_const":
                        self.emit(
                            f"{a}pub const <{ty}>::{m['name']}: "
                            f"{self.rtype(m['inner']['assoc_const']['type'])}"
                        )
                    elif mk == "assoc_type":
                        self.emit(f"{a}pub type <{ty}>::{m['name']}")
            else:
                syn = " [auto]" if im["is_synthetic"] else ""
                self.emit(
                    f"impl{head} {neg}{self.rpath(trait)} for {self.rtype(for_ty)}{tail}{syn}"
                )


def listing_for(json_path: Path) -> list[str]:
    import json

    doc = json.loads(json_path.read_text(encoding="utf-8"))
    return Lister(doc).list_all()


def snapshot_text(lines: list[str]) -> str:
    return HEADER + "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="regenerate snapshots")
    ap.add_argument("--crate", action="append", help="limit to a crate (repeatable)")
    ap.add_argument("--json-dir", type=Path, help="reuse rustdoc JSON from this directory")
    ap.add_argument("--target-dir", type=Path, help="CARGO_TARGET_DIR for the doc build")
    args = ap.parse_args()

    crates = published_crates()
    if args.crate:
        unknown = sorted(set(args.crate) - set(crates))
        if unknown:
            print(f"unknown or unpublished crate(s): {unknown}", file=sys.stderr)
            return 2
        crates = sorted(args.crate)
    doc_dir = args.json_dir or run_rustdoc(crates, args.target_dir)

    failed = False
    for name in crates:
        json_path = doc_dir / f"{name.replace('-', '_')}.json"
        if not json_path.exists():
            print(f"missing rustdoc JSON for {name}: {json_path}", file=sys.stderr)
            return 2
        text = snapshot_text(listing_for(json_path))
        snap = SNAPSHOT_DIR / f"{name}.txt"
        if args.write:
            SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            snap.write_text(text, encoding="utf-8", newline="\n")
            print(f"wrote {snap.relative_to(ROOT)} ({text.count(chr(10)) - 3} items)")
            continue
        if not snap.exists():
            print(f"FAIL {name}: no snapshot {snap.relative_to(ROOT)}; run with --write")
            failed = True
            continue
        old = snap.read_text(encoding="utf-8")
        if old != text:
            failed = True
            print(f"FAIL {name}: public API differs from {snap.relative_to(ROOT)}")
            diff = difflib.unified_diff(
                old.splitlines(), text.splitlines(), "snapshot", "current", lineterm="", n=0
            )
            for i, line in enumerate(diff):
                if i >= 200:
                    print("  ... (diff truncated)")
                    break
                print("  " + line)
        else:
            print(f"ok   {name}")
    if args.write:
        stale = sorted(
            p for p in SNAPSHOT_DIR.glob("*.txt") if p.stem not in set(published_crates())
        )
        for p in stale:
            p.unlink()
            print(f"removed stale {p.relative_to(ROOT)}")
        return 0
    if failed:
        print(
            "\nPublic API changed. If intended, regenerate and commit the snapshot diff:\n"
            "  python3 scripts/gate_public_api.py --write",
            file=sys.stderr,
        )
        return 1
    print("Public API gate PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
