"""Content-addressed evidence objects, each stored once.

An object is bytes addressed by their SHA-256: a manifest, one of its fields, an input
entry, the input addresses of one directory, the directories of one manifest, or an item
table.
Shard `<aa>.txt` holds the objects whose address starts with `aa`, one
`<address> <bytes>` line each, so 256 files hold the whole store. Writers only append, under
a lock on the shard; `compact` rewrites the shards sorted, keeping the objects named.

A shard is self-verifying: every line's bytes hash to its address. A run's appended lines
are therefore checked by content rather than by remembering the file's hash.

`encode` stores a JSON value as a tree of shared objects: a value longer than `LIMIT` bytes
becomes `{"@": address}`, and a long list is cut into chunks where an item's hash says so,
`{"@chunks": [address, ...]}`. Cuts depend only on the items, so two lists that share a run
of items share its chunks: passes whose inputs, reviews or guard tables mostly agree store
only what differs.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

OBJECTS = "parity/evidence_objects"
HEX = re.compile(r"[0-9a-f]{64}\Z")
SHARD = re.compile(r"parity/evidence_objects/[0-9a-f]{2}\.txt\Z")


# Values at most this long stay inline; a list cut into chunks averages `CHUNK` items each.
LIMIT = 512
CHUNK = 16
REF, CHUNKS = "@", "@chunks"


class CorruptObject(ValueError):
    """An object is missing, or its bytes do not match its address."""


def address_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def shard_name(address: str) -> str:
    return f"{OBJECTS}/{address[:2]}.txt"


def parse(data: bytes, name: str = "shard") -> dict[str, bytes]:
    """The objects of shard bytes `data`, by address; `CorruptObject` on any bad line."""
    if data and not data.endswith(b"\n"):
        raise CorruptObject(f"{name} does not end in a newline")
    objects: dict[str, bytes] = {}
    for number, line in enumerate(data.split(b"\n")[:-1], 1):
        address, _, body = line.partition(b" ")
        text = address.decode("ascii", "replace")
        if not HEX.fullmatch(text) or address_of(body) != text:
            raise CorruptObject(f"{name} line {number} does not match its address")
        if objects.setdefault(text, body) != body:
            raise CorruptObject(f"{name} names {text} twice")
    return objects


class Store:
    """The objects under `root`. Reads load a shard once and pick up later appends."""

    def __init__(self, root: Path):
        self.root = root
        self.shards: dict[str, tuple[int, dict[str, bytes]]] = {}

    def _path(self, prefix: str) -> Path:
        return self.root / OBJECTS / f"{prefix}.txt"

    def _objects(self, prefix: str, handle=None) -> dict[str, bytes]:
        path = self._path(prefix)
        size, objects = self.shards.get(prefix, (0, {}))
        current = os.fstat(handle.fileno()).st_size if handle else (
            path.stat().st_size if path.is_file() else 0)
        if current < size:
            size, objects = 0, {}
        if current > size:
            with open(path, "rb") as file:
                file.seek(size)
                tail = file.read(current - size)
            # Appends land whole lines under the shard lock; a reader without it stops at the
            # last complete line and reads the rest next time.
            if handle is None and not tail.endswith(b"\n"):
                tail = tail[: tail.rfind(b"\n") + 1]
            objects = {**objects, **parse(tail, path.name)}
            size += len(tail)
            self.shards[prefix] = (size, objects)
        return objects

    def get(self, address: str) -> bytes:
        if not isinstance(address, str) or not HEX.fullmatch(address):
            raise CorruptObject(f"invalid object address {address!r}")
        cached = self.shards.get(address[:2])
        data = cached[1].get(address) if cached else None
        if data is None:
            data = self._objects(address[:2]).get(address)
        if data is None:
            raise CorruptObject(f"missing evidence object {address}")
        return data

    def has(self, address: str) -> bool:
        if not isinstance(address, str) or not HEX.fullmatch(address):
            return False
        cached = self.shards.get(address[:2])
        return (cached is not None and address in cached[1]) or address in self._objects(address[:2])

    def addresses(self) -> set[str]:
        return {address for prefix in self.prefixes() for address in self._objects(prefix)}

    def prefixes(self) -> list[str]:
        base = self.root / OBJECTS
        return sorted(p.stem for p in base.glob("??.txt")) if base.is_dir() else []

    def put(self, objects: dict[str, bytes]) -> list[str]:
        """Store each `address -> bytes` not yet stored; the shards written, repo-relative."""
        by_prefix: dict[str, dict[str, bytes]] = {}
        for address, data in objects.items():
            if address_of(data) != address or b"\n" in data:
                raise CorruptObject(f"object {address} is not storable under its address")
            by_prefix.setdefault(address[:2], {})[address] = data
        written = []
        (self.root / OBJECTS).mkdir(parents=True, exist_ok=True)
        for prefix, wanted in sorted(by_prefix.items()):
            if all(self.has(address) for address in wanted):
                continue
            with open(self._path(prefix), "ab") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                stored = self._objects(prefix, handle)
                new = b"".join(address.encode() + b" " + data + b"\n"
                               for address, data in sorted(wanted.items()) if address not in stored)
                if new:
                    handle.write(new)
                    handle.flush()
                    os.fsync(handle.fileno())
                    written.append(shard_name(prefix))
        return written

    def compact(self, keep: set[str]) -> list[str]:
        """Rewrite every shard sorted, with only the objects in `keep`; the addresses dropped."""
        dropped: list[str] = []
        for prefix in self.prefixes():
            path = self._path(prefix)
            with open(path, "rb+") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                objects = self._objects(prefix, handle)
                kept = {a: d for a, d in objects.items() if a in keep}
                dropped += sorted(set(objects) - set(kept))
                if not kept:
                    path.unlink()
                    self.shards.pop(prefix, None)
                    continue
                data = b"".join(a.encode() + b" " + kept[a] + b"\n" for a in sorted(kept))
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
                    file.write(data)
                os.replace(file.name, path)
                self.shards[prefix] = (len(data), kept)
        return dropped


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _is_marker(value) -> bool:
    return isinstance(value, dict) and len(value) == 1 and next(iter(value)) in (REF, CHUNKS)


def encode(value, objects: dict[str, bytes], *, encoded_parts: bool = False):
    """`value` with every long part replaced by a reference; the parts are added to `objects`.
    With `encoded_parts`, references in `value` are parts `encode` already returned."""
    return _encode(value, objects, encoded_parts)[0]


def encode_object(value, objects: dict[str, bytes]) -> dict:
    """`value` encoded and always stored as its own object: a reference to it."""
    encoded, data = _encode(value, objects, False)
    return encoded if isinstance(encoded, dict) and REF in encoded else {REF: _put(data, objects)}


def _encode(value, objects: dict[str, bytes], encoded_parts: bool) -> tuple[object, bytes]:
    """The encoded value and its canonical bytes, built from its parts' bytes."""
    if isinstance(value, (dict, list)):
        data = _canonical(value)
        # A short value is stored inline as it is, unless it holds a reference marker.
        if len(data) <= LIMIT and (b'{"@' not in data or encoded_parts):
            return value, data
    if isinstance(value, dict):
        if _is_marker(value):
            if encoded_parts:
                return value, _canonical(value)
            raise ValueError("a stored value cannot look like an object reference")
        parts = {key: _encode(item, objects, encoded_parts) for key, item in value.items()}
        encoded = {key: part[0] for key, part in parts.items()}
        data = b"{" + b",".join(_canonical(key) + b":" + parts[key][1] for key in sorted(parts)) + b"}"
    elif isinstance(value, list):
        parts = [_encode(item, objects, encoded_parts) for item in value]
        encoded = [part[0] for part in parts]
        data = b"[" + b",".join(part[1] for part in parts) + b"]"
        if len(parts) > CHUNK and len(data) > LIMIT:
            chunks, current = [], []
            for item, item_data in parts:
                current.append(item_data)
                if int(address_of(item_data)[:8], 16) % CHUNK == 0:
                    chunks.append(_put(b"[" + b",".join(current) + b"]", objects))
                    current = []
            if current:
                chunks.append(_put(b"[" + b",".join(current) + b"]", objects))
            encoded = {CHUNKS: chunks}
            data = _canonical(encoded)
    else:
        return value, _canonical(value)
    if len(data) > LIMIT:
        encoded = {REF: _put(data, objects)}
        data = _canonical(encoded)
    return encoded, data


def _put(data: bytes, objects: dict[str, bytes]) -> str:
    address = address_of(data)
    objects[address] = data
    return address


def decode(value, read):
    """The value `encode` stored; `read` returns an object's decoded value (`Decoded`)."""
    if isinstance(value, dict):
        if _is_marker(value):
            if REF in value:
                return read(_address(value[REF]))
            chunks = value[CHUNKS]
            if not isinstance(chunks, list):
                raise CorruptObject("invalid chunk list")
            items = []
            for address in chunks:
                chunk = read(_address(address))
                if not isinstance(chunk, list):
                    raise CorruptObject("invalid chunk")
                items.extend(chunk)
            return items
        return {key: decode(item, read) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item, read) for item in value]
    return value


def _address(address) -> str:
    if not isinstance(address, str) or not HEX.fullmatch(address):
        raise CorruptObject(f"invalid object reference {address!r}")
    return address


class Decoded:
    """Each object's decoded value, decoded once: passes share their objects, so a reader of
    many passes decodes each shared part once. The values are shared; callers do not mutate
    them."""

    def __init__(self, read_bytes):
        self.read_bytes = read_bytes
        self.values: dict[str, object] = {}

    def __call__(self, address: str):
        value = self.values.get(address)
        if value is None:
            data = self.read_bytes(address)
            try:
                value = json.loads(data)
            except ValueError as error:
                raise CorruptObject(f"object {address} is not JSON") from error
            # An object without a reference is its own decoded value.
            if b'{"@' in data:
                value = decode(value, self)
            self.values[address] = value
        return value


_NAMED = re.compile(rb'"(?:@|table)":"([0-9a-f]{64})"|"@chunks":\[([^\]]*)\]')
_ADDRESS = re.compile(rb"[0-9a-f]{64}")


def reachable(store: "Store", address: str) -> set[str]:
    """`address` and every stored object its bytes name, transitively: references, and the
    item tables input entries name by address."""
    reached, todo = set(), [address]
    while todo:
        current = todo.pop()
        if current in reached or not store.has(current):
            continue
        reached.add(current)
        for one, chunks in _NAMED.findall(store.get(current)):
            todo.extend(a.decode() for a in ([one] if one else _ADDRESS.findall(chunks)))
    return reached


@contextlib.contextmanager
def locked(root: Path):
    """Hold the store's lock: writers of a file that indexes objects take it."""
    base = root / OBJECTS
    base.mkdir(parents=True, exist_ok=True)
    handle = os.open(base, os.O_RDONLY)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    finally:
        os.close(handle)


def write(root: Path, objects: dict[str, bytes]) -> list[str]:
    """Store `objects` under `root`; the shards written. Callers reach it through this module,
    so a run that defers its writes can replace it."""
    return Store(root).put(objects)


def appended_only(root: Path, name: str) -> bool:
    """Whether the working shard `name` is its committed bytes plus whole, valid lines."""
    if not SHARD.fullmatch(name):
        return False
    path = root / name
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        return False
    data = path.read_bytes()
    committed = subprocess.run(["git", "cat-file", "blob", f"HEAD:{name}"], cwd=root,
                               capture_output=True).stdout
    if not data.startswith(committed):
        return False
    try:
        parse(data[len(committed):], name)
    except CorruptObject:
        return False
    return True
