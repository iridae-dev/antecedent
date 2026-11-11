"""Item-level Rust evidence: which items a record depends on, and whether they changed.

Earning starts from the items coverage shows executed and follows the compiler
inventory (`tools/evidence-analyzer/driver.rs`): an executed function's whole text,
the signature of every function it names (its whole text, transitively, when the
coverage is a sample of the runs: `callee_bodies`), the whole text of every type, constant,
static, trait, alias and macro it uses (and theirs, transitively), the header of
each impl it runs in, and the `mod` declarations that put each file in its crate.
Name resolution and method selection are guarded rather than recorded:

* a *scope* guard keeps what decides how a used name resolves in a module: items
  and imports of that name, glob imports and item macros (and, where a method is
  resolved, every import and trait declared there);
* a *probe* guard keeps the inherent items of the method's name on the receiver's
  types, and the impls of the traits in scope that could apply to them;
* a *macro* guard keeps every `macro_rules!` of a used macro's name.

A record whose executed code observes source positions (`file!`, `line!`,
`column!`, `Location`, panic hooks or backtraces) keeps its files whole: any edit
that moves a line could change what it observes. Positions are otherwise not
inputs: they reach a binary only through panic locations and debug information,
which a passing assertion does not read.

Storage: a file's item table (its locators in order) is content-addressed in
`parity/evidence_items/`, shared by every record and file version with that
structure; a manifest entry names the table and selects items by index. A file
whose bytes are unchanged is current without being parsed.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from rust_items import File, ParseError, canonical, impl_parts, scope_entries

STORE = "parity/evidence_items"
PARTS = ("full", "sig", "head", "inner")
_POSITION_MACRO = re.compile(r"\b(?:file|line|column)!\s*[(\[{]")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------- parse cache

_PARSED: dict[tuple, object] = {}


def parse(path: Path) -> File:
    """The parsed file, once per file state. Raises `ParseError` (or `OSError`)."""
    from evidence_manifests import _stat_signature
    signature = _stat_signature(path)
    key = (str(path), signature)
    if signature is None or key not in _PARSED:
        try:
            value: object = File(path.read_text(encoding="utf-8"))
        except (ParseError, UnicodeDecodeError) as error:
            value = ParseError(str(error))
        if signature is None:
            if isinstance(value, Exception):
                raise value
            return value  # type: ignore[return-value]
        _PARSED[key] = value
    value = _PARSED[key]
    if isinstance(value, Exception):
        raise value
    return value  # type: ignore[return-value]


# ---------------------------------------------------------------------- name filter

class Names:
    """A Bloom filter of identifiers. A false positive only adds a guard entry."""

    HASHES = 7

    def __init__(self, bits: int, data: bytes):
        self.bits = bits
        self.data = bytearray(data)

    @classmethod
    def of(cls, names: set[str]) -> Names:
        bits = max(1024, 12 * len(names))
        bits += (-bits) % 8
        filter_ = cls(bits, bytes(bits // 8))
        for name in names:
            for index in filter_._indices(name):
                filter_.data[index // 8] |= 1 << (index % 8)
        return filter_

    @classmethod
    def load(cls, entry: dict) -> Names:
        return cls(int(entry["bits"]), bytes.fromhex(entry["filter"]))

    def _indices(self, name: str):
        digest = hashlib.sha256(name.encode()).digest()
        for i in range(self.HASHES):
            yield int.from_bytes(digest[4 * i:4 * i + 4], "big") % self.bits

    def __contains__(self, name: str) -> bool:
        return all(self.data[i // 8] >> (i % 8) & 1 for i in self._indices(name))

    def entry(self) -> dict:
        return {"bits": self.bits, "filter": bytes(self.data).hex()}


# ---------------------------------------------------------------------- inventory

@dataclass
class Inventory:
    items: dict[str, dict] = field(default_factory=dict)
    modules: dict[str, dict] = field(default_factory=dict)
    globs: dict[str, set[str]] = field(default_factory=dict)
    by_file: dict[str, list[str]] = field(default_factory=dict)
    file_modules: dict[str, str] = field(default_factory=dict)
    node_index: dict[str, tuple] = field(default_factory=dict)

    @classmethod
    def read(cls, lines) -> Inventory:
        inventory = cls()
        for line in lines:
            if not line.strip():
                continue
            record = json.loads(line)
            kind = record["t"]
            if kind == "item":
                merged = inventory.items.get(record["key"])
                if merged is None:
                    merged = inventory.items[record["key"]] = {
                        "kind": record["kind"], "mod": record["mod"], "deps": set(), "bdeps": set(),
                        "lookups": set(), "probes": set(), "observes": False, "includes": False}
                merged["deps"].update(map(tuple, record["deps"]))
                merged["bdeps"].update(map(tuple, record["bdeps"]))
                merged["lookups"].update(map(tuple, record["lookups"]))
                merged["probes"].update(json.dumps(p, sort_keys=True) for p in record["probes"])
                merged["observes"] |= record["observes"]
                merged["includes"] |= record["includes"]
            elif kind == "mod":
                inventory.modules.setdefault(record["key"], record)
            elif kind == "glob":
                inventory.globs.setdefault(record["mod"], set()).add(record["src"])
            else:
                raise ValueError(f"unknown compiler inventory record {kind!r}")
        for key in inventory.items:
            inventory.by_file.setdefault(key.rsplit(":", 1)[0], []).append(key)
        for key, module in inventory.modules.items():
            if not module["chain"]:
                inventory.file_modules.setdefault(module["file"], key)
        return inventory


_INVENTORIES: dict[tuple, Inventory] = {}


def load_inventory(directory: Path) -> Inventory:
    from evidence_manifests import _stat_signature
    files = sorted(directory.glob("*.jsonl"))
    key = (str(directory), tuple(_stat_signature(f) for f in files))
    if key not in _INVENTORIES:
        def lines():
            for path in files:
                yield from path.read_text(encoding="utf-8").splitlines()
        _INVENTORIES[key] = Inventory.read(lines())
    return _INVENTORIES[key]


# ---------------------------------------------------------------------- selection

def crate_dir(root: Path, rel: str) -> str:
    """The nearest directory holding a Cargo.toml, relative to `root`."""
    path = (root / rel).parent
    while path != root and not (path / "Cargo.toml").is_file():
        path = path.parent
    return path.relative_to(root).as_posix() if path != root else "."


class Selection:
    """The items one record depends on, grown from its executed positions."""

    def __init__(self, root: Path, inventory: Inventory, *, callee_bodies: bool = False):
        self.root = root
        self.inventory = inventory
        # Coverage from a sample of the runs (a calibration slice): a function that
        # executed code names may run in a replicate the sample did not, so it is kept
        # whole, transitively, rather than by its signature.
        self.callee_bodies = callee_bodies
        self.parts: dict[str, dict[str, set[str]]] = {}
        self.whole: dict[str, str] = {}
        self.names: set[str] = set()
        self.scopes: dict[str, bool] = {}
        self.probes: set[str] = set()
        self.macros: set[str] = set()
        self.observes = False
        self.reasons: list[str] = []
        self._key_nodes: dict[str, tuple[str, object] | None] = {}
        self._chained: set[str] = set()

    # --- files and nodes

    def file(self, rel: str) -> File | None:
        if rel in self.whole:
            return None
        try:
            return parse(self.root / rel)
        except (ParseError, OSError) as error:
            self.keep_whole(rel, f"unparsed: {error}")
            return None

    def keep_whole(self, rel: str, reason: str) -> None:
        if rel not in self.whole:
            self.whole[rel] = reason
            self.parts.pop(rel, None)

    def node_items(self, rel: str, file: File) -> dict[str, list[dict]]:
        """The inventory items of each node of `rel`, shared by every selection of one inventory."""
        cache = self.inventory.node_index
        if cache.get(rel, (None,))[0] is not file:
            index: dict[str, list[dict]] = {}
            for key in self.inventory.by_file.get(rel, []):
                node = file.innermost(file.char_offset(int(key.rsplit(":", 1)[1])))
                index.setdefault(node.locator, []).append(self.inventory.items[key])
            cache[rel] = (file, index)
        return cache[rel][1]

    def key_node(self, key: str):
        if key not in self._key_nodes:
            rel, offset = key.rsplit(":", 1)
            file = self.file(rel)
            self._key_nodes[key] = None if file is None else (rel, file.innermost(file.char_offset(int(offset))))
        return self._key_nodes[key]

    # --- growth

    def execute(self, rel: str, offset: int) -> None:
        """An executed position (character offset)."""
        file = self.file(rel)
        if file is None:
            return
        node = file.innermost(offset)
        if node is file.root:
            self.keep_whole(rel, "executed code outside any item")
            return
        self.need(rel, node, "full")

    def need(self, rel: str, node, part: str) -> None:
        file = self.file(rel)
        if file is None:
            return
        work = [(node, part)]
        while work:
            node, part = work.pop()
            if node is file.root:
                part = "inner"
            have = self.parts.setdefault(rel, {}).setdefault(node.locator, set())
            if part in have or ("full" in have and part != "inner"):
                continue
            have.add(part)
            if node is file.root:
                continue
            if part == "full" and _POSITION_MACRO.search(file.part(node, "full")):
                self.observes = True
            # Enclosing items decide what this one is (cfg, generics, the impl's type).
            parent = node.parent
            if parent is not None:
                if parent.kind == "mod":
                    work.append((parent, "head"))
                    if parent.inner:
                        work.append((parent, "inner"))
                elif parent.kind == "impl":
                    work.append((parent, "head"))
                elif parent.kind in ("trait", "fn", "macro", "foreign"):
                    work.append((parent, "full"))
                elif parent is file.root and parent.inner:
                    work.append((parent, "inner"))
            if node.kind == "macro_rules":
                self.macros.add(node.name)
            self.chain(rel)
            for item in self.node_items(rel, file).get(node.locator, []):
                deps = item["deps"] | item["bdeps"] if part == "full" else item["deps"]
                for module, name in item["lookups"]:
                    self.names.add(name)
                    self.scopes.setdefault(module, False)
                if part == "full":
                    if any(json.loads(p)["early"] for p in item["probes"]):
                        # An earlier probe step could find a newly imported trait.
                        self.scopes[item["mod"]] = True
                    self.probes.update(item["probes"])
                    if item["observes"]:
                        self.observes = True
                for relation, key in deps:
                    target = self.key_node(key)
                    if target is None:
                        continue
                    target_rel, target_node = target
                    if relation == "fn" and self.callee_bodies:
                        wanted = "full"
                    elif relation == "fn":
                        target_file = self.file(target_rel)
                        header = target_file.part(target_node, "sig") if target_file else ""
                        wanted = "full" if re.search(r"\bimpl\b|\basync\b", header) else "sig"
                        wanted = wanted if target_node.kind == "fn" else "full"
                    elif relation == "head":
                        wanted = "head" if target_node.kind == "impl" else "full"
                    else:
                        wanted = "full"
                    if target_rel == rel:
                        work.append((target_node, wanted))
                    else:
                        self.need(target_rel, target_node, wanted)
            if rel in self.whole:
                return

    def chain(self, rel: str) -> None:
        """The `mod` declarations, up to the crate root, that compile this file."""
        if rel in self._chained:
            return
        self._chained.add(rel)
        module = self.inventory.file_modules.get(rel)
        if module is None:
            self.keep_whole(rel, "no compiler module for this file")
            return
        parent_key = self.inventory.modules[module]["parent"]
        if parent_key is None:
            return
        parent = self.inventory.modules.get(parent_key)
        name = self.inventory.modules[module]["name"]
        parent_file = self.file(parent["file"]) if parent else None
        container = parent_file.module(parent["chain"]) if parent_file else None
        found = [c for c in (container.children if container else []) if c.kind == "mod" and c.name == name]
        if len(found) != 1:
            if parent:
                self.keep_whole(parent["file"], f"module declaration of {rel} not found")
            return
        self.need(parent["file"], found[0], "full")

    # --- output

    def entries(self, checker, crate_dirs: list[str], scope_dir: str) -> tuple[list[dict], dict[str, bytes]]:
        """Manifest inputs and the item tables they name (by address)."""
        tables: dict[str, bytes] = {}
        inputs: list[dict] = []
        if self.observes:
            for rel in list(self.parts):
                self.keep_whole(rel, "executed code observes source positions")
        names = Names.of(self.names)
        scope_files: dict[str, list[list]] = {}
        for module_key, traits in sorted(self._scope_closure().items()):
            module = self.inventory.modules.get(module_key)
            if module is None:
                continue
            scope_files.setdefault(module["file"], []).append([module["chain"], traits])
        for rel in sorted(set(self.parts) | set(scope_files)):
            if rel in self.whole:
                continue
            file = self.file(rel)
            if file is None:
                continue
            table = list(file.by_locator)
            position = {locator: i for i, locator in enumerate(table)}
            select = {}
            for part in PARTS:
                chosen = sorted(position[loc] for loc, parts in self.parts.get(rel, {}).items()
                                if part in parts and loc in position
                                and (part in ("full", "inner") or "full" not in parts))
                if part == "inner" and "" in self.parts.get(rel, {}) and "inner" in self.parts[rel][""]:
                    chosen = [-1] + chosen
                if chosen:
                    select[part] = chosen
            data = canonical(table)
            address = sha(data)
            tables[address] = data
            entry = {"kind": "rust_items", "path": rel, "role": "compile",
                     "file": checker.input_digest("file", rel), "table": address,
                     "select": select, "scopes": scope_files.get(rel, [])}
            entry["digest"] = items_digest(file, table, entry, names)
            inputs.append(entry)
        for rel, reason in sorted(self.whole.items()):
            if (self.root / rel).is_file():
                inputs.append({"kind": "evidence_file", "path": rel, "role": "compile",
                               "digest": checker.input_digest("evidence_file", rel)})
            self.reasons.append(f"{rel}: {reason}")
        inputs.append({"kind": "rust_names", "path": scope_dir, "role": "selection",
                       **names.entry(), "digest": sha(canonical(names.entry()))})
        spec = self.guard_spec(crate_dirs)
        guard = {"kind": "rust_guards", "path": scope_dir, "role": "selection", "dirs": sorted(crate_dirs),
                 "spec": spec, "tree": guard_tree(checker, crate_dirs)}
        guard["digest"] = sha(canonical(guard_results(self.root, checker, spec, crate_dirs)))
        inputs.append(guard)
        return inputs, tables

    def _scope_closure(self) -> dict[str, bool]:
        """Scope guards, through glob imports and the paths of the imports they keep."""
        from rust_items import use_bindings
        scopes = dict(self.scopes)
        work = list(scopes)
        followed: set[tuple[str, str]] = set()
        while work:
            module = work.pop()
            for source in self.inventory.globs.get(module, ()):
                traits = scopes[module] or scopes.get(source, False)
                if source not in scopes or (traits and not scopes[source]):
                    scopes[source] = traits
                    work.append(source)
            record = self.inventory.modules.get(module)
            file = self.file(record["file"]) if record else None
            container = file.module(record["chain"]) if file else None
            if container is None:
                continue
            for child in container.children:
                if child.kind != "use" or (record["file"], child.locator) in followed:
                    continue
                bound, glob = use_bindings(file.part(child, "full"))
                if not (glob or scopes[module] or bound & self.names):
                    continue
                followed.add((record["file"], child.locator))
                for item in self.node_items(record["file"], file).get(child.locator, []):
                    for target, name in item["lookups"]:
                        self.names.add(name)
                        if target not in scopes:
                            scopes[target] = False
                            work.append(target)
        return scopes

    def guard_spec(self, crate_dirs: list[str]) -> dict:
        inherent, impls = set(), set()
        for text in self.probes:
            probe = json.loads(text)
            heads = {name for _, name in probe["heads"]}
            for key, name in probe["heads"]:
                if key:
                    inherent.add((probe["name"], name, crate_dir(self.root, key.rsplit(":", 1)[0])))
            if not probe["early"]:
                # Found at the first step: only a same-named inherent item preempts it.
                continue
            concrete = sorted(heads - {"*"})
            for _, trait in probe["resolved"]:
                if concrete:
                    impls.add((trait, tuple(concrete), False))
            for _, trait in probe["traits"]:
                impls.add((trait, tuple(sorted(heads)), "*" in heads))
        return {"inherent": sorted(map(list, inherent)), "impls": sorted([t, list(h), o] for t, h, o in impls),
                "macros": sorted(self.macros)}


# ---------------------------------------------------------------------- digests

def items_digest(file: File, table: list[str], entry: dict, names) -> str:
    """The digest of an entry's selected parts and scopes in `file`; `KeyError` if gone."""
    rows = []
    for part in PARTS:
        for index in entry["select"].get(part, []):
            locator = "" if index == -1 else table[index]
            digest = file.digest(locator, part)
            if digest is None:
                raise KeyError(f"{locator} is gone")
            rows.append([locator, part, digest])
    for chain, traits in entry["scopes"]:
        module = file.module(chain)
        if module is None:
            raise KeyError(f"module {'::'.join(chain)} is gone")
        rows.append(["scope", chain, traits, scope_entries(file, module, names, traits)])
    return sha(canonical(rows))


def _rust_files(checker, dirs: list[str]) -> list[str]:
    files = []
    for directory in dirs:
        files.extend(rel for rel in checker.members(directory) if rel.endswith(".rs"))
    return sorted(set(files))


def guard_tree(checker, dirs: list[str]) -> str:
    # A checker reads one state of the tree: many records share the same directories.
    cache = checker.__dict__.setdefault("_guard_trees", {})
    key = tuple(dirs)
    if key not in cache:
        cache[key] = sha(canonical([[rel, checker.input_digest("file", rel)] for rel in _rust_files(checker, dirs)]))
    return cache[key]


_FACTS: dict[tuple, dict] = {}
# Facts are a pure function of a file's bytes: kept across processes, keyed by content.
_FACTS_VERSION = 1
_DISK_FACTS: dict[str, dict] | None = None
_DISK_DIRTY = [False]


def _facts_cache_path(root: Path) -> Path:
    return root / ".cache" / "evidence-items" / f"facts-v{_FACTS_VERSION}.json"


def _disk_facts(root: Path) -> dict[str, dict]:
    global _DISK_FACTS
    if _DISK_FACTS is None:
        try:
            _DISK_FACTS = json.loads(_facts_cache_path(root).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _DISK_FACTS = {}
    return _DISK_FACTS


def save_facts(root: Path) -> None:
    """Write the content-keyed facts cache, if this process added to it."""
    if not _DISK_DIRTY[0] or _DISK_FACTS is None:
        return
    path = _facts_cache_path(root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(_DISK_FACTS, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
        _DISK_DIRTY[0] = False
    except OSError:
        pass


def _facts(root: Path, checker, rel: str) -> dict:
    """A file's inherent items, trait impls, macros and alias names, once per file state."""
    digest = checker.input_digest("file", rel)
    key = (rel, digest)
    if key not in _FACTS:
        stored = _disk_facts(root).get(digest) if isinstance(digest, str) else None
        if stored is not None:
            facts = {"aliases": set(stored["aliases"]), "unreadable": stored["unreadable"],
                     **{k: [tuple(x) for x in stored[k]] for k in ("inherent", "impls", "macros")}}
        else:
            facts = _parse_facts(root, rel, digest)
            if isinstance(digest, str):
                _disk_facts(root)[digest] = {**facts, "aliases": sorted(facts["aliases"])}
                _DISK_DIRTY[0] = True
        _FACTS[key] = facts
    return _FACTS[key]


def _parse_facts(root: Path, rel: str, digest) -> dict:
    text = (root / rel).read_text(encoding="utf-8", errors="replace")
    facts = {"aliases": set(re.findall(r"\btype\s+([A-Za-z_][A-Za-z0-9_]*)", text)),
             "inherent": [], "impls": [], "macros": [], "unreadable": None}
    try:
        file = parse(root / rel)
    except ParseError:
        facts["unreadable"] = digest
    else:
        for node in _walk(file.root):
            if node.kind == "impl":
                trait, head, _ = impl_parts(node.name)
                if trait is None:
                    facts["inherent"].extend((head, c.name, c.locator) for c in node.children)
                else:
                    facts["impls"].append((trait, head, node.locator))
            elif node.kind == "macro_rules":
                facts["macros"].append((node.name, node.locator, sha(file.part(node, "full").encode())))
    return facts


def _guard_index(root: Path, checker, dirs: list[str]) -> dict:
    """The facts of every Rust file under `dirs`, indexed by name; once per checker and `dirs`."""
    cache = checker.__dict__.setdefault("_guard_indexes", {})
    key = tuple(dirs)
    if key not in cache:
        index = {"aliases": {}, "unreadable": [], "inherent": {}, "impls": {}, "macros": {}}
        for rel in _rust_files(checker, dirs):
            facts = _facts(root, checker, rel)
            directory = next((d for d in dirs if rel.startswith(d.rstrip("/") + "/")), ".")
            index["aliases"].setdefault(directory, set()).update(facts["aliases"])
            if facts["unreadable"] is not None:
                index["unreadable"].append([rel, "file", facts["unreadable"]])
                continue
            for head, name, locator in facts["inherent"]:
                index["inherent"].setdefault(name, []).append((rel, head, locator))
            for trait, head, locator in facts["impls"]:
                index["impls"].setdefault(trait, []).append((rel, head, locator))
            for name, locator, digest in facts["macros"]:
                index["macros"].setdefault(name, []).append([rel, locator, digest])
        index["any_alias"] = set().union(*index["aliases"].values()) if index["aliases"] else set()
        cache[key] = index
        save_facts(root)
    return cache[key]


def guard_results(root: Path, checker, spec: dict, dirs: list[str]) -> list:
    """What each guard keeps: the inherent items, impls and macros it found."""
    inherent: dict[tuple, list] = {tuple(g): [] for g in spec["inherent"]}
    impls: dict[tuple, list] = {(t, tuple(h), o): [] for t, h, o in spec["impls"]}
    macros: dict[str, list] = {m: [] for m in spec["macros"]}
    if not (inherent or impls or macros):
        return [[], [], []]
    index = _guard_index(root, checker, dirs)
    aliases, any_alias = index["aliases"], index["any_alias"]
    # An unreadable file may hold anything: every guard keeps its bytes.
    for bucket in (*inherent.values(), *impls.values(), *macros.values()):
        bucket.extend(index["unreadable"])
    for (method, type_name, directory), bucket in inherent.items():
        prefix = directory.rstrip("/") + "/"
        for rel, head, locator in index["inherent"].get(method, ()):
            if directory != "." and not rel.startswith(prefix):
                continue
            if head == type_name or head == "*" or head in aliases.get(directory, ()):
                bucket.append([rel, locator])
    for (name, heads, open_), bucket in impls.items():
        wanted = set(heads)
        for rel, head, locator in index["impls"].get(name, ()):
            if open_ or head in wanted or head == "*" or head in any_alias:
                bucket.append([rel, locator])
    for name, bucket in macros.items():
        bucket.extend(index["macros"].get(name, ()))
    return [sorted([list(k), sorted(v)] for k, v in inherent.items()),
            sorted([[k[0], list(k[1]), k[2]], sorted(v)] for k, v in impls.items()),
            sorted([k, sorted(v)] for k, v in macros.items())]


def _walk(node):
    for child in node.children:
        yield child
        yield from _walk(child)


# ---------------------------------------------------------------------- checking

class ItemChecker:
    """Currency of `rust_items`, `rust_names` and `rust_guards` entries."""

    def __init__(self, checker):
        self.checker = checker
        self.tables: dict[str, list[str]] = {}
        self.cache: dict[tuple, bool] = {}
        # Tables of a manifest being published, before they are written.
        self.pending: dict[str, bytes] = {}

    def table(self, address: str) -> list[str]:
        if address not in self.tables:
            data = self.pending.get(address) or self.checker.path(f"{STORE}/{address}.json").read_bytes()
            if sha(data) != address:
                raise ValueError("item table content does not match its address")
            table = json.loads(data)
            if not isinstance(table, list) or not all(isinstance(t, str) for t in table):
                raise ValueError("invalid item table")
            self.tables[address] = table
        return self.tables[address]

    def items_current(self, entry: dict, names_entry: dict | None) -> bool:
        rel = entry["path"]
        if self.checker.input_digest("file", rel) == entry["file"]:
            # Unchanged bytes: the table must still exist for later checks.
            self.table(entry["table"])
            return True
        if names_entry is None:
            return False
        key = ("items", rel, self.checker.input_digest("file", rel), entry["digest"], names_entry["filter"])
        if key not in self.cache:
            try:
                file = parse(self.checker.path(rel))
                self.cache[key] = items_digest(file, self.table(entry["table"]), entry,
                                               Names.load(names_entry)) == entry["digest"]
            except (ParseError, KeyError, IndexError, OSError):
                self.cache[key] = False
        return self.cache[key]

    def guards_current(self, entry: dict) -> bool:
        dirs = entry["dirs"]
        if guard_tree(self.checker, dirs) == entry["tree"]:
            return True
        key = ("guards", entry["digest"], canonical(entry["spec"]), tuple(dirs))
        if key not in self.cache:
            results = guard_results(self.checker.root, self.checker, entry["spec"], dirs)
            self.cache[key] = sha(canonical(results)) == entry["digest"]
        return self.cache[key]
