#!/usr/bin/env python3
"""Clean-room consumer of the Antecedent interface-contract, evidence-ledger and
composition-bundle formats.

This file imports nothing from `antecedent` and nothing outside the Python
standard library. It implements, from the written specification in `spec/`
only:

* the artifact container (magic, manifest, section table, per-section BLAKE3);
* a minimal CBOR decoder (RFC 8949, definite and indefinite lengths);
* BLAKE3 (the reference algorithm, default hash mode, 32-byte output), checked
  against published vectors before any identity is trusted;
* the canonical identity writer (length-framed words and texts under a domain
  tag) and the identity recomputation of each artifact kind;
* a small JSON Schema (draft 2020-12 subset) validator for the three schemas;
* the structural and semantic checks the specification states.

Usage:

    validate.py [--spec-dir DIR] [--fixtures DIR]      run the conformance manifest
    validate.py --kind KIND FILE [--retained-identity HEX]
                                                       validate one artifact
    validate.py --self-test [--fixtures DIR]           check the decoder and BLAKE3
    validate.py --dump-tables                          print the vocabulary tables as JSON

Exit status: 0 when every expectation holds (or the single artifact is valid),
1 when an artifact is rejected or an expectation fails, 2 on a usage error.

SPDX-License-Identifier: MIT OR Apache-2.0
"""

import json
import math
import os
import re
import struct
import sys

# ---------------------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------------------

CLASSES = (
    "container", "checksum", "kind", "unsupported", "version", "feature",
    "required_feature", "schema", "structure", "semantic", "identity",
)


class Reject(Exception):
    """An artifact that must be refused; `klass` is one of `CLASSES`."""

    def __init__(self, klass, message):
        Exception.__init__(self, "%s: %s" % (klass, message))
        self.klass = klass
        self.message = message


# ---------------------------------------------------------------------------------------
# BLAKE3 (reference algorithm, unkeyed hash, 32-byte output)
# ---------------------------------------------------------------------------------------

_IV = (0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
       0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19)
_PERM = (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8)
_CHUNK_START, _CHUNK_END, _PARENT, _ROOT = 1, 2, 4, 8
_M = 0xFFFFFFFF


def _ror(x, n):
    return ((x >> n) | (x << (32 - n))) & _M


def _g(s, a, b, c, d, mx, my):
    s[a] = (s[a] + s[b] + mx) & _M
    s[d] = _ror(s[d] ^ s[a], 16)
    s[c] = (s[c] + s[d]) & _M
    s[b] = _ror(s[b] ^ s[c], 12)
    s[a] = (s[a] + s[b] + my) & _M
    s[d] = _ror(s[d] ^ s[a], 8)
    s[c] = (s[c] + s[d]) & _M
    s[b] = _ror(s[b] ^ s[c], 7)


def _round(s, m):
    _g(s, 0, 4, 8, 12, m[0], m[1])
    _g(s, 1, 5, 9, 13, m[2], m[3])
    _g(s, 2, 6, 10, 14, m[4], m[5])
    _g(s, 3, 7, 11, 15, m[6], m[7])
    _g(s, 0, 5, 10, 15, m[8], m[9])
    _g(s, 1, 6, 11, 12, m[10], m[11])
    _g(s, 2, 7, 8, 13, m[12], m[13])
    _g(s, 3, 4, 9, 14, m[14], m[15])


def _compress(cv, words, counter, block_len, flags):
    s = list(cv[:8]) + list(_IV[:4]) + [
        counter & _M, (counter >> 32) & _M, block_len, flags]
    block = list(words)
    for index in range(7):
        _round(s, block)
        if index < 6:
            block = [block[p] for p in _PERM]
    for index in range(8):
        s[index] ^= s[index + 8]
        s[index + 8] ^= cv[index]
    return s


def _block_words(block):
    padded = block + b"\x00" * (64 - len(block))
    return struct.unpack("<16I", padded)


def _chunk_output(chunk, counter):
    """(input chaining value, block words, counter, block length, flags) of the
    last block of a chunk, after compressing every earlier block."""
    blocks = [chunk[i:i + 64] for i in range(0, len(chunk), 64)] or [b""]
    cv = list(_IV)
    for index, block in enumerate(blocks[:-1]):
        flags = _CHUNK_START if index == 0 else 0
        cv = _compress(cv, _block_words(block), counter, 64, flags)[:8]
    last = blocks[-1]
    flags = _CHUNK_END | (_CHUNK_START if len(blocks) == 1 else 0)
    return (cv, _block_words(last), counter, len(last), flags)


def _chaining_value(output):
    cv, words, counter, block_len, flags = output
    return _compress(cv, words, counter, block_len, flags)[:8]


def _subtree(chunks, first_counter):
    if len(chunks) == 1:
        return _chunk_output(chunks[0], first_counter)
    left_count = 1 << ((len(chunks) - 1).bit_length() - 1)
    left = _chaining_value(_subtree(chunks[:left_count], first_counter))
    right = _chaining_value(_subtree(chunks[left_count:], first_counter + left_count))
    return (list(_IV), tuple(left + right), 0, 64, _PARENT)


def blake3(data):
    """The 32-byte BLAKE3 hash of `data`."""
    chunks = [data[i:i + 1024] for i in range(0, len(data), 1024)] or [b""]
    cv, words, _counter, block_len, flags = _subtree(chunks, 0)
    out = _compress(cv, words, 0, block_len, flags | _ROOT)
    return struct.pack("<8I", *out[:8])


def blake3_hex(data):
    return blake3(data).hex()


# ---------------------------------------------------------------------------------------
# CBOR (RFC 8949), decode only
# ---------------------------------------------------------------------------------------

class CborError(Exception):
    pass


class _Break(object):
    pass


_BREAK = _Break()


class _Reader(object):
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def take(self, count):
        if count < 0 or self.pos + count > len(self.data):
            raise CborError("truncated CBOR item")
        chunk = self.data[self.pos:self.pos + count]
        self.pos += count
        return chunk


def _argument(reader, info):
    if info < 24:
        return info
    if info == 24:
        return reader.take(1)[0]
    if info == 25:
        return struct.unpack(">H", reader.take(2))[0]
    if info == 26:
        return struct.unpack(">I", reader.take(4))[0]
    if info == 27:
        return struct.unpack(">Q", reader.take(8))[0]
    raise CborError("invalid additional information %d" % info)


def _item(reader, depth):
    if depth > 64:
        raise CborError("CBOR nesting too deep")
    initial = reader.take(1)[0]
    major, info = initial >> 5, initial & 0x1F
    if major == 7:
        if info == 20:
            return False
        if info == 21:
            return True
        if info in (22, 23):
            return None
        if info == 25:
            return struct.unpack(">e", reader.take(2))[0]
        if info == 26:
            return struct.unpack(">f", reader.take(4))[0]
        if info == 27:
            return struct.unpack(">d", reader.take(8))[0]
        if info == 31:
            return _BREAK
        raise CborError("unsupported simple value %d" % info)
    if info == 31:
        return _indefinite(reader, major, depth)
    value = _argument(reader, info)
    if major == 0:
        return value
    if major == 1:
        return -1 - value
    if major == 2:
        return bytes(reader.take(value))
    if major == 3:
        try:
            return reader.take(value).decode("utf-8")
        except UnicodeDecodeError:
            raise CborError("text string is not UTF-8")
    if major == 4:
        return [_item_value(reader, depth) for _ in range(value)]
    if major == 5:
        return _map(reader, value, depth)
    if major == 6:
        return _item_value(reader, depth)  # tags are transparent
    raise CborError("unreachable major type")


def _item_value(reader, depth):
    value = _item(reader, depth + 1)
    if value is _BREAK:
        raise CborError("unexpected break")
    return value


def _map(reader, count, depth):
    result = {}
    for _ in range(count):
        key = _item_value(reader, depth)
        if not isinstance(key, str):
            raise CborError("map key is not a text string")
        if key in result:
            raise CborError("duplicate map key %r" % key)
        result[key] = _item_value(reader, depth)
    return result


def _indefinite(reader, major, depth):
    if major in (2, 3):
        parts = []
        while True:
            part = _item(reader, depth + 1)
            if part is _BREAK:
                break
            parts.append(part)
        if major == 2:
            return b"".join(parts)
        return "".join(parts)
    if major == 4:
        items = []
        while True:
            part = _item(reader, depth + 1)
            if part is _BREAK:
                return items
            items.append(part)
    if major == 5:
        result = {}
        while True:
            key = _item(reader, depth + 1)
            if key is _BREAK:
                return result
            if not isinstance(key, str):
                raise CborError("map key is not a text string")
            if key in result:
                raise CborError("duplicate map key %r" % key)
            result[key] = _item_value(reader, depth)
    raise CborError("invalid indefinite length")


def cbor_decode(data):
    """Decode exactly one CBOR item; trailing bytes are refused."""
    reader = _Reader(data)
    value = _item_value(reader, 0)
    if reader.pos != len(data):
        raise CborError("%d trailing bytes after the CBOR item" % (len(data) - reader.pos))
    return value


# ---------------------------------------------------------------------------------------
# Canonical identity writer
# ---------------------------------------------------------------------------------------

class Writer(object):
    """Length-framed canonical stream hashed with BLAKE3.

    A word is a u64 little-endian; text and bytes are a u64 little-endian
    length then the raw bytes; the first field is the domain tag as text.
    """

    def __init__(self, tag):
        self.buf = bytearray()
        self.text(tag)

    def word(self, value):
        self.buf += struct.pack("<Q", value)
        return self

    def bytes_(self, raw):
        self.word(len(raw))
        self.buf += raw
        return self

    def text(self, value):
        return self.bytes_(value.encode("utf-8"))

    def texts(self, values):
        self.word(len(values))
        for value in values:
            self.text(value)
        return self

    def real(self, value):
        return self.word(struct.unpack("<Q", struct.pack("<d", value))[0])

    def hex(self):
        return blake3_hex(bytes(self.buf))


# ---------------------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------------------

MAGIC = b"CAUSAL\x00\x00"
MAX_FILE_BYTES = 256 * 1024 * 1024


class Container(object):
    def __init__(self, manifest, sections):
        self.manifest = manifest
        self.sections = sections  # ordered list of (id, bytes)

    def kind(self):
        kind = self.manifest.get("artifact_kind")
        if isinstance(kind, dict) and list(kind.keys()) == ["other"]:
            return kind["other"]
        return None

    def section(self, section_id):
        for name, payload in self.sections:
            if name == section_id:
                return payload
        return None


def parse_container(data):
    if len(data) > MAX_FILE_BYTES:
        raise Reject("container", "file is over the consumer bound")
    if len(data) < 16 or data[:8] != MAGIC:
        raise Reject("container", "bad magic")
    version, manifest_len = struct.unpack("<II", data[8:16])
    if version != 1:
        raise Reject("version", "unsupported container version %d" % version)
    if 16 + manifest_len > len(data):
        raise Reject("container", "truncated manifest")
    try:
        manifest = cbor_decode(data[16:16 + manifest_len])
    except CborError as error:
        raise Reject("container", "manifest: %s" % error)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("sections"), list):
        raise Reject("container", "manifest has no section table")
    position = 16 + manifest_len
    sections = []
    seen = set()
    for descriptor in manifest["sections"]:
        if not isinstance(descriptor, dict):
            raise Reject("container", "section descriptor is not a map")
        section_id = descriptor.get("id")
        if not isinstance(section_id, str) or section_id in seen:
            raise Reject("container", "section id missing or repeated")
        seen.add(section_id)
        if position + 4 > len(data):
            raise Reject("container", "truncated section length")
        (length,) = struct.unpack("<I", data[position:position + 4])
        position += 4
        if descriptor.get("compressed_size") != length:
            raise Reject("container", "section %s: size does not match its descriptor" % section_id)
        if position + length > len(data):
            raise Reject("container", "section %s: truncated payload" % section_id)
        payload = data[position:position + length]
        position += length
        stored = descriptor.get("blake3")
        if isinstance(stored, list) and all(isinstance(b, int) and 0 <= b < 256 for b in stored):
            stored = bytes(stored)
        if not isinstance(stored, bytes) or len(stored) != 32:
            raise Reject("container", "section %s: no 32-byte checksum" % section_id)
        if blake3(payload) != stored:
            raise Reject("checksum", "section %s: BLAKE3 checksum does not match" % section_id)
        if descriptor.get("compression") is not None:
            raise Reject("unsupported",
                         "section %s uses compression %r, which this consumer does not decode"
                         % (section_id, descriptor.get("compression")))
        if descriptor.get("uncompressed_size") != length:
            raise Reject("container", "section %s: uncompressed size differs" % section_id)
        sections.append((section_id, payload))
    if position != len(data):
        raise Reject("container", "%d bytes after the last section" % (len(data) - position))
    return Container(manifest, sections)


# ---------------------------------------------------------------------------------------
# JSON Schema (draft 2020-12 subset)
# ---------------------------------------------------------------------------------------

def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    return (isinstance(value, (int, float))) and not isinstance(value, bool)


def _type_ok(value, name):
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "null":
        return value is None
    if name == "integer":
        return _is_int(value)
    if name == "number":
        return _is_number(value)
    raise ValueError("unsupported schema type %r" % name)


def _canonical(value):
    if isinstance(value, bool):
        return "b:%s" % value
    if _is_int(value):
        return "i:%d" % value
    if isinstance(value, float):
        return "f:%r" % value
    if isinstance(value, str):
        return "s:" + json.dumps(value)
    if value is None:
        return "n"
    if isinstance(value, list):
        return "[" + ",".join(_canonical(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join("%s=%s" % (json.dumps(k), _canonical(value[k]))
                              for k in sorted(value)) + "}"
    return repr(value)


def _resolve(root, reference):
    if reference == "#":
        return root
    if not reference.startswith("#/"):
        raise ValueError("unsupported $ref %r" % reference)
    node = root
    for part in reference[2:].split("/"):
        node = node[part.replace("~1", "/").replace("~0", "~")]
    return node


def schema_errors(instance, schema, root, path="$"):
    """The list of violations of `schema` by `instance` (empty when valid)."""
    if schema is True:
        return []
    if schema is False:
        return ["%s: nothing is allowed here" % path]
    errors = []
    if "$ref" in schema:
        errors += schema_errors(instance, _resolve(root, schema["$ref"]), root, path)
    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_type_ok(instance, n) for n in names):
            return errors + ["%s: expected %s" % (path, "|".join(names))]
    if "const" in schema and _canonical(instance) != _canonical(schema["const"]):
        errors.append("%s: expected the constant %r" % (path, schema["const"]))
    if "enum" in schema and _canonical(instance) not in [_canonical(v) for v in schema["enum"]]:
        errors.append("%s: %r is not one of %r" % (path, instance, schema["enum"]))
    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append("%s: shorter than %d" % (path, schema["minLength"]))
        if "maxLength" in schema and len(instance.encode("utf-8")) > schema["maxLength"]:
            errors.append("%s: longer than %d bytes" % (path, schema["maxLength"]))
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            errors.append("%s: does not match %s" % (path, schema["pattern"]))
    if _is_number(instance):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append("%s: below %r" % (path, schema["minimum"]))
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append("%s: above %r" % (path, schema["maximum"]))
    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append("%s: fewer than %d items" % (path, schema["minItems"]))
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append("%s: more than %d items" % (path, schema["maxItems"]))
        if schema.get("uniqueItems"):
            keys = [_canonical(v) for v in instance]
            if len(set(keys)) != len(keys):
                errors.append("%s: items are not unique" % path)
        prefix = schema.get("prefixItems", [])
        for index, sub in enumerate(prefix):
            if index < len(instance):
                errors += schema_errors(instance[index], sub, root, "%s[%d]" % (path, index))
        if "items" in schema:
            for index in range(len(prefix), len(instance)):
                errors += schema_errors(instance[index], schema["items"], root,
                                        "%s[%d]" % (path, index))
    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errors.append("%s: missing required field %r" % (path, key))
        properties = schema.get("properties", {})
        for key, value in instance.items():
            if key in properties:
                errors += schema_errors(value, properties[key], root, "%s.%s" % (path, key))
            else:
                extra = schema.get("additionalProperties", True)
                if extra is False:
                    errors.append("%s: unknown field %r" % (path, key))
                elif extra is not True:
                    errors += schema_errors(value, extra, root, "%s.%s" % (path, key))
    if "oneOf" in schema:
        matches = [sub for sub in schema["oneOf"] if not schema_errors(instance, sub, root, path)]
        if len(matches) != 1:
            errors.append("%s: matches %d of the alternatives, expected exactly one"
                          % (path, len(matches)))
    if "anyOf" in schema:
        if not any(not schema_errors(instance, sub, root, path) for sub in schema["anyOf"]):
            errors.append("%s: matches none of the alternatives" % path)
    if "allOf" in schema:
        for sub in schema["allOf"]:
            errors += schema_errors(instance, sub, root, path)
    return errors


# ---------------------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------------------

HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _expect(condition, klass, message):
    if not condition:
        raise Reject(klass, message)


def _strictly_increasing(values):
    return all(values[i] < values[i + 1] for i in range(len(values) - 1))


def _bits_to_float(bits):
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def _load_meta(container, kind, section_id):
    expected = {"other": kind}
    _expect(container.manifest.get("artifact_kind") == expected, "kind",
            "artifact kind is %r, expected %r" % (container.manifest.get("artifact_kind"), kind))
    _expect([name for name, _ in container.sections][:1] == [section_id], "container",
            "the first section is not %r" % section_id)
    try:
        return cbor_decode(container.section(section_id))
    except CborError as error:
        raise Reject("container", "%s: %s" % (section_id, error))


def _gate_version(meta, feature, supported_version=1):
    _expect(isinstance(meta, dict), "schema", "metadata is not a map")
    version = meta.get("version")
    _expect(_is_int(version), "schema", "metadata has no integer version")
    _expect(version == supported_version, "version",
            "version %d is not the supported version %d" % (version, supported_version))
    if feature is not None:
        _expect(meta.get("feature") == feature, "feature",
                "feature marker %r is not %r" % (meta.get("feature"), feature))


def _check_schema(instance, schema_path):
    with open(schema_path, "r") as handle:
        schema = json.load(handle)
    errors = schema_errors(instance, schema, schema)
    if errors:
        raise Reject("schema", "; ".join(errors[:5]) + (" (+%d more)" % (len(errors) - 5)
                                                         if len(errors) > 5 else ""))


# ---------------------------------------------------------------------------------------
# interface_contract_v1
# ---------------------------------------------------------------------------------------

OP_TAG = {"point": 1, "mean": 2, "law": 3, "draws": 4, "joint_draws": 5,
          "covariance": 6, "interval": 7, "posterior": 8}
UNCERTAINTY_OPS = {"law", "draws", "joint_draws", "covariance", "interval", "posterior"}
MEANING_PARADIGM = {
    "parameter_posterior": "posterior", "causal_functional_posterior": "posterior",
    "posterior_predictive": "posterior", "interventional_predictive": "posterior",
    "estimator_sampling": "sampling", "bootstrap": "resampling", "empirical_outcome": "empirical",
}
ROLE_TAG = {"treatment": 1, "outcome": 2, "covariate": 3, "mediator": 4, "selection": 5,
            "utility": 6}
CALIBRATION_TAG = {"none": 1, "point_only": 2, "unmeasured": 3, "measured": 4, "inherited": 5}
CALIBRATION_REQUIREMENT_TAG = {"not_required": 0, "measured": 1}
DEPENDENCE_TAG = {"unconstrained": 0, "declared_evidence": 1, "disjoint_evidence": 2}
SUPPORT_TAG = {"real": 1, "non_negative": 2, "binary": 3, "interval": 4, "categorical": 5}
MAX_TEXT_BYTES = 4096
INTERFACE_CONTRACT_TAG = "antecedent.interface_contract"
INTERFACE_CONTRACT_ARTIFACT_TAG = "antecedent.interface_contract.artifact"


def _nonblank(text):
    return isinstance(text, str) and text.strip() != "" and len(text.encode("utf-8")) <= MAX_TEXT_BYTES


def _write_quantity(writer, q):
    writer.text(q["variable_id"]).text(q["variable_name"]).text(q["units"])
    writer.text(q["population_id"]).text(q["regime_id"]).text(q["functional_id"])
    writer.text(q["transform_id"])
    writer.word(ROLE_TAG[q["role"]]).word(q["horizon"]).word(len(q["conditioning"]))
    for condition in q["conditioning"]:
        writer.text(condition["variable_id"]).text(condition["value_id"])


def _write_support(writer, support):
    writer.word(SUPPORT_TAG[support["kind"]])
    if support["kind"] == "interval":
        writer.word(support["lower_bits"]).word(support["upper_bits"])
    elif support["kind"] == "categorical":
        writer.texts(support["levels"])


def _write_uncertainty(writer, uncertainty):
    if uncertainty is None:
        writer.word(0)
    else:
        writer.word(1).text(uncertainty["meaning"]).text(uncertainty["paradigm"])


def _write_operations(writer, operations):
    writer.word(len(operations))
    for operation in operations:
        writer.word(OP_TAG[operation])


def interface_contract_identity(contract):
    """The contract identity (hex) of a decoded `contract` map."""
    writer = Writer(INTERFACE_CONTRACT_TAG)
    writer.word(1).text(contract["component"])
    writer.word(len(contract["produces"]))
    for port in contract["produces"]:
        writer.text(port["name"])
        _write_quantity(writer, port["quantity"])
        writer.text(port["semantic_identity"])
        _write_support(writer, port["support"])
        _write_operations(writer, port["operations"])
        _write_uncertainty(writer, port["uncertainty"])
        calibration = port["calibration"]
        writer.word(CALIBRATION_TAG[calibration["status"]])
        if calibration["status"] == "measured":
            writer.text(calibration["record_id"])
        elif calibration["status"] == "inherited":
            writer.text(calibration["input"])
        writer.texts(port["measurement_models"])
        evidence = port["evidence"]
        if evidence["declared"]:
            writer.word(1).word(len(evidence["sources"]))
            for source in evidence["sources"]:
                writer.text(source)
        else:
            writer.word(0)
        writer.texts(port["provenance"])
    writer.word(len(contract["consumes"]))
    for port in contract["consumes"]:
        writer.text(port["name"])
        _write_quantity(writer, port["quantity"])
        writer.text(port["semantic_identity"])
        _write_support(writer, port["support"])
        _write_operations(writer, port["operations"])
        _write_uncertainty(writer, port["uncertainty"])
        writer.word(CALIBRATION_REQUIREMENT_TAG[port["calibration"]])
        writer.texts(port["measurement_models"])
        writer.word(DEPENDENCE_TAG[port["dependence"]])
        writer.texts(port["provenance"])
    return writer.hex()


def interface_contract_artifact_identity(meta):
    writer = Writer(INTERFACE_CONTRACT_ARTIFACT_TAG)
    writer.word(meta["version"]).text(meta["feature"]).texts(meta["required_features"])
    writer.text(meta["contract_identity"])
    return writer.hex()


def _check_quantity(q, where):
    for field in ("variable_id", "variable_name", "units", "population_id", "regime_id",
                  "functional_id", "transform_id"):
        _expect(_nonblank(q[field]), "semantic", "%s: quantity field %s is blank" % (where, field))
    ids = [c["variable_id"] for c in q["conditioning"]]
    for condition in q["conditioning"]:
        _expect(_nonblank(condition["variable_id"]) and _nonblank(condition["value_id"]),
                "semantic", "%s: a condition is blank" % where)
    _expect(_strictly_increasing(ids), "semantic",
            "%s: conditioning is not sorted by variable and distinct" % where)


def _check_support(support, where):
    kind = support["kind"]
    if kind == "interval":
        lower = _bits_to_float(support["lower_bits"])
        upper = _bits_to_float(support["upper_bits"])
        _expect(math.isfinite(lower) and math.isfinite(upper) and lower <= upper, "semantic",
                "%s: interval support bounds are not finite and ordered" % where)
    elif kind == "categorical":
        levels = support["levels"]
        _expect(all(_nonblank(level) for level in levels), "semantic",
                "%s: a support level is blank" % where)
        _expect(_strictly_increasing(levels), "structure",
                "%s: support levels are not sorted and distinct" % where)


def _check_operations_and_uncertainty(port, where):
    operations = port["operations"]
    tags = [OP_TAG[o] for o in operations]
    _expect(_strictly_increasing(tags), "structure",
            "%s: operations are not in canonical order" % where)
    uncertainty = port["uncertainty"]
    needs = any(o in UNCERTAINTY_OPS for o in operations)
    if uncertainty is not None:
        _expect(MEANING_PARADIGM[uncertainty["meaning"]] == uncertainty["paradigm"], "semantic",
                "%s: the stated paradigm is not the meaning's paradigm" % where)
    _expect(needs == (uncertainty is not None), "semantic",
            "%s: an uncertainty meaning is stated exactly when an uncertainty-bearing operation is declared" % where)
    if "posterior" in operations and uncertainty is not None:
        _expect(uncertainty["paradigm"] == "posterior", "semantic",
                "%s: a posterior operation needs a posterior meaning" % where)


def _check_sets(port, where):
    for field in ("measurement_models", "provenance"):
        _expect(_strictly_increasing(port[field]), "structure",
                "%s: %s are not sorted and distinct" % (where, field))


def _check_interface_contract(meta):
    contract = meta["contract"]
    _expect(_nonblank(contract["component"]), "semantic", "component is blank")
    for field in ("produces", "consumes"):
        names = [p["name"] for p in contract[field]]
        _expect(_strictly_increasing(names), "structure",
                "%s are not sorted by name and distinct" % field)
        for name in names:
            _expect(_nonblank(name), "semantic", "a port name is blank")
    consumed = set(p["name"] for p in contract["consumes"])
    for port in contract["consumes"]:
        where = "consumes.%s" % port["name"]
        _check_quantity(port["quantity"], where)
        _expect(_nonblank(port["semantic_identity"]), "semantic", "%s: blank semantic identity" % where)
        _check_support(port["support"], where)
        _check_operations_and_uncertainty(port, where)
        _check_sets(port, where)
        if port["calibration"] == "measured":
            _expect("interval" in port["operations"] or "posterior" in port["operations"],
                    "semantic", "%s: measured calibration relies on an interval or a posterior" % where)
    for port in contract["produces"]:
        where = "produces.%s" % port["name"]
        _check_quantity(port["quantity"], where)
        _expect(_nonblank(port["semantic_identity"]), "semantic", "%s: blank semantic identity" % where)
        _check_support(port["support"], where)
        _check_operations_and_uncertainty(port, where)
        _check_sets(port, where)
        evidence = port["evidence"]
        _expect(_strictly_increasing(evidence["sources"]), "structure",
                "%s: evidence sources are not sorted and distinct" % where)
        calibration = port["calibration"]
        status = calibration["status"]
        if status == "point_only":
            _expect(not any(o in UNCERTAINTY_OPS for o in port["operations"]), "semantic",
                    "%s: point-only beside an uncertainty statement" % where)
        elif status == "measured":
            _expect(_nonblank(calibration["record_id"]), "semantic",
                    "%s: a measured label names its record" % where)
            _expect("interval" in port["operations"] or "posterior" in port["operations"],
                    "semantic", "%s: measured calibration relies on an interval or a posterior" % where)
        elif status == "inherited":
            _expect(calibration["input"] in consumed, "semantic",
                    "%s: inherits from a port the component does not consume" % where)


def check_interface_contract(container, spec_dir, retained_identity=None):
    meta = _load_meta(container, "interface_contract_v1", "interface_contract_meta")
    _gate_version(meta, "interface_contract_v1")
    required = meta.get("required_features")
    _expect(isinstance(required, list), "schema", "required_features is not an array")
    _expect(len(required) == 0, "required_feature",
            "required features %r are not defined by this version" % (required,))
    _check_schema(meta, os.path.join(spec_dir, "interface-contract.schema.json"))
    _check_interface_contract(meta)
    contract_identity = interface_contract_identity(meta["contract"])
    _expect(meta["contract_identity"] == contract_identity, "identity",
            "contract identity %s is not the recomputed %s" % (meta["contract_identity"], contract_identity))
    identity = interface_contract_artifact_identity(meta)
    _expect(meta["identity"] == identity, "identity",
            "artifact identity %s is not the recomputed %s" % (meta["identity"], identity))
    if retained_identity is not None:
        _expect(retained_identity == identity, "identity",
                "the artifact identity is not the retained identity")
    return {"identity": identity, "contract_identity": contract_identity, "meta": meta}


# ---------------------------------------------------------------------------------------
# evidence_ledger_v1
# ---------------------------------------------------------------------------------------

EVIDENCE_LEDGER_TAG = "antecedent.evidence_ledger.artifact"
SYNTHESIS_REPLAY_TOLERANCE = 1e-12


def evidence_ledger_identity(meta):
    writer = Writer(EVIDENCE_LEDGER_TAG)
    writer.word(meta["version"]).text(meta["feature"]).text(meta["head"])
    writer.text(meta["state_digest"])
    writer.word(len(meta["operations"]))
    for operation in meta["operations"]:
        writer.text(operation["kind"]).text(operation["id"])
    writer.word(len(meta["syntheses"]))
    for synthesis in meta["syntheses"]:
        writer.text(synthesis["result"]).text(synthesis["method"]).texts(synthesis["sources"])
        for field in ("estimates", "variances", "covariance", "tau_support",
                      "tau_probabilities", "values"):
            writer.word(len(synthesis[field]))
            for value in synthesis[field]:
                writer.word(value)
        for field in ("rho_lo", "rho_hi", "prior_mean", "prior_variance"):
            if synthesis[field] is None:
                writer.word(0)
            else:
                writer.word(1).word(synthesis[field])
    return writer.hex()


def _recompute_inverse_variance(estimates, variances):
    # Same arithmetic and summation order as the reference: precision p_i = 1/v_i,
    # total = sum p_i, normalized weights w_i = p_i / total, estimate = sum w_i y_i,
    # Q = sum p_i (y_i - estimate)^2, variance = 1 / total; values are
    # [estimate, variance, Q, w_1 .. w_n].
    precision = [1.0 / v for v in variances]
    total = 0.0
    for p in precision:
        total += p
    weights = [p / total for p in precision]
    estimate = 0.0
    for w, y in zip(weights, estimates):
        estimate += w * y
    q = 0.0
    for p, y in zip(precision, estimates):
        q += p * (y - estimate) * (y - estimate)
    return [estimate, 1.0 / total, q] + weights


def _close(stored, recomputed):
    if math.isnan(stored) or math.isnan(recomputed):
        return math.isnan(stored) and math.isnan(recomputed)
    return abs(stored - recomputed) <= SYNTHESIS_REPLAY_TOLERANCE * max(abs(recomputed), 1.0)


def check_evidence_ledger(container, spec_dir, retained_identity=None):
    meta = _load_meta(container, "evidence_ledger_v1", "evidence_ledger_meta")
    _gate_version(meta, "evidence_ledger_v1")
    _check_schema(meta, os.path.join(spec_dir, "evidence-ledger.schema.json"))
    operations = meta["operations"]
    if operations:
        _expect(operations[-1]["id"] == meta["head"], "structure",
                "the head is not the identity of the last operation")
    created_sources = set()
    created_results = set()
    for operation in operations:
        effect = operation["effect"]
        for field in ("created_source",):
            if effect[field] is not None:
                created_sources.add(effect[field])
        if effect["created_result"] is not None:
            created_results.add(effect["created_result"])
    identifiers = [op["id"] for op in operations]
    _expect(len(set(identifiers)) == len(identifiers), "structure", "an operation id is repeated")
    recomputed_methods = 0
    for synthesis in meta["syntheses"]:
        _expect(synthesis["result"] in created_results, "structure",
                "a synthesis is bound to a result no operation created")
        _expect(all(s in created_sources for s in synthesis["sources"]), "structure",
                "a synthesis names a source no operation created")
        if synthesis["method"] == "fixed_effect_inverse_variance":
            estimates = [_bits_to_float(b) for b in synthesis["estimates"]]
            variances = [_bits_to_float(b) for b in synthesis["variances"]]
            _expect(len(estimates) == len(variances) == len(synthesis["sources"]) > 0
                    and all(v > 0 and math.isfinite(v) for v in variances), "semantic",
                    "inverse-variance inputs are not positive finite variances, one per study")
            expected = _recompute_inverse_variance(estimates, variances)
            stored = [_bits_to_float(b) for b in synthesis["values"]]
            _expect(len(stored) == len(expected)
                    and all(_close(s, e) for s, e in zip(stored, expected)), "semantic",
                    "the stored synthesis values are not the recomputed pooled values")
            recomputed_methods += 1
    identity = evidence_ledger_identity(meta)
    _expect(meta["identity"] == identity, "identity",
            "artifact identity %s is not the recomputed %s" % (meta["identity"], identity))
    if retained_identity is not None:
        _expect(retained_identity == identity, "identity",
                "the artifact identity is not the retained identity")
    return {"identity": identity, "meta": meta, "recomputed_syntheses": recomputed_methods}


# ---------------------------------------------------------------------------------------
# composition_bundle_v1
# ---------------------------------------------------------------------------------------

NODE_TAG = "antecedent.composition_bundle.node.v1"
BUNDLE_TAG = "antecedent.composition_bundle.v1"
LINK_TAG = "antecedent.composition_provenance.link.v1"
KIND_STAGE = {
    "causal_contract": "causal_contract",
    "execution_or_fit": "distribution_artifact",
    "distribution": "distribution_artifact",
    "frozen_scores": "distribution_artifact",
    "external_claim": "external_provider",
    "attestation": "external_provider",
    "evidence_relationship": "evidence",
    "quantity_coordinates": "evidence",
    "support_trust_calibration": "evidence",
    "repair_report": "evidence",
    "recalculation_receipt": "evidence",
    "transformation": "transformation",
    "rollout": "transformation",
    "decision_contract": "decision_contract",
    "decision_result": "claim",
    "inverse_query": "claim",
    "sensitivity": "sensitivity_input",
    "study_ranking": "study_ranking_provider",
}


def _requires_canonical(requires):
    if "provider" in requires:
        p = requires["provider"]
        return "provider|%s|%s|%s" % (p["provider_id"], p["snapshot_id"], p["request_fingerprint"])
    d = requires["data"]
    return "data|%s|%s" % (d["snapshot_id"], d["digest"])


def _node_digest(node, container):
    writer = Writer(NODE_TAG)
    writer.text(node["id"]).text(node["kind"]).text(node["identity"])
    if node["embedded"]:
        payload = container.section("embedded." + node["id"])
        writer.text("embedded").text(blake3_hex(payload))
    else:
        writer.text("reference").text(node["identity"])
        writer.text(_requires_canonical(node["requires"]))
        facts = node["facts"]
        writer.word(len(facts))
        for key in sorted(facts):
            writer.text(key).text(facts[key])
        writer.word(len(node["inspected"]))
        for key, value in node["inspected"]:
            writer.text(key).real(float(value))
    return writer.hex()


def composition_bundle_digests(body, container):
    """(node digests by id, link digests by id, bundle identity) of a decoded body."""
    nodes = body["nodes"]
    by_id = {}
    for node in nodes:
        by_id[node["id"]] = node
    dependencies = sorted((e["from"], e["to"]) for e in body["edges"])
    parents = dict((node["id"], []) for node in nodes)
    for upstream, dependent in dependencies:
        _expect(upstream in by_id and dependent in by_id, "structure", "an edge names an unknown node")
        _expect(upstream != dependent, "structure", "a node depends on itself")
        parents[dependent].append(upstream)
    node_digests = dict((n["id"], _node_digest(n, container)) for n in nodes)
    link_digests = {}
    visiting = set()

    def link(node_id):
        if node_id in link_digests:
            return link_digests[node_id]
        _expect(node_id not in visiting, "structure", "the dependency graph has a cycle")
        visiting.add(node_id)
        parent_digests = [link(p) for p in parents[node_id]]
        visiting.discard(node_id)
        link_id = "%s@%s" % (node_id, node_digests[node_id])
        writer = Writer(LINK_TAG)
        writer.text(link_id).text(KIND_STAGE[by_id[node_id]["kind"]]).texts(parent_digests)
        link_digests[node_id] = writer.hex()
        return link_digests[node_id]

    for node in nodes:
        link(node["id"])
    writer = Writer(BUNDLE_TAG)
    writer.word(len(nodes))
    for node in nodes:
        writer.text(node["id"]).text(node_digests[node["id"]]).text(link_digests[node["id"]])
    writer.word(len(dependencies))
    for upstream, dependent in dependencies:
        writer.text(upstream).text(dependent).text(link_digests[upstream])
    return node_digests, link_digests, writer.hex(), dependencies


def check_composition_bundle(container, spec_dir, retained_identity=None):
    body = _load_meta(container, "composition_bundle_v1", "bundle.body")
    _gate_version(body, None)
    _check_schema(body, os.path.join(spec_dir, "composition-bundle.schema.json"))
    nodes = body["nodes"]
    ids = [n["id"] for n in nodes]
    _expect(_strictly_increasing(ids), "structure", "nodes are not sorted by id and distinct")
    embedded = [n["id"] for n in nodes if n["embedded"]]
    _expect(len(container.sections) == len(embedded) + 1, "structure",
            "the container has sections that no node accounts for")
    for node in nodes:
        _expect(_nonblank(node["id"]) and _nonblank(node["identity"]), "semantic",
                "a node has a blank id or identity")
        if node["embedded"]:
            _expect(container.section("embedded." + node["id"]) is not None, "structure",
                    "an embedded node has no content section")
            _expect(node["requires"] is None and not node["facts"] and not node["inspected"],
                    "structure", "an embedded node carries reference fields")
        else:
            _expect(node["requires"] is not None, "structure",
                    "a reference node states no requirement")
    edges = body["edges"]
    pairs = [(e["from"], e["to"]) for e in edges]
    _expect(_strictly_increasing(pairs), "structure", "edges are not sorted and distinct")
    node_digests, link_digests, identity, _ = composition_bundle_digests(body, container)
    for edge in edges:
        _expect(edge["upstream_digest"] == link_digests[edge["from"]], "identity",
                "edge %s -> %s carries a digest that is not its upstream's Merkle digest"
                % (edge["from"], edge["to"]))
    _expect(body["identity"] == identity, "identity",
            "bundle identity %s is not the recomputed %s" % (body["identity"], identity))
    if retained_identity is not None:
        _expect(retained_identity == identity, "identity",
                "the bundle identity is not the retained identity")
    return {"identity": identity, "meta": body, "node_digests": node_digests}


CHECKS = {
    "interface_contract_v1": check_interface_contract,
    "evidence_ledger_v1": check_evidence_ledger,
    "composition_bundle_v1": check_composition_bundle,
}


def validate_bytes(data, kind, spec_dir, retained_identity=None):
    """Validate one artifact; raise `Reject` or return the check result."""
    if kind not in CHECKS:
        raise Reject("kind", "unknown artifact kind %r" % kind)
    container = parse_container(data)
    return CHECKS[kind](container, spec_dir, retained_identity)


# ---------------------------------------------------------------------------------------
# Self tests and the conformance manifest
# ---------------------------------------------------------------------------------------

KNOWN_BLAKE3 = (
    (0, "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"),
    (1, "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213"),
    (2, "7b7015bb92cf0b318037702a6cdd81dee41224f734684c2c122cd6359cb1ee63"),
    (1023, "10108970eeda3eb932baac1428c7a2163b0e924c9a9e25b35bba72b28f70bd11"),
    (1024, "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7"),
    (1025, "d00278ae47eb27b34faecf67b4fe263f82d5412916c1ffd97c8cb7fb814b8444"),
)


def _pattern(length):
    return bytes(i % 251 for i in range(length))


CBOR_VECTORS = (
    ("00", 0), ("17", 23), ("1818", 24), ("1864", 100), ("1903e8", 1000),
    ("1a000f4240", 1000000), ("1b000000e8d4a51000", 1000000000000), ("20", -1), ("3863", -100),
    ("f4", False), ("f5", True), ("f6", None), ("6161", "a"), ("60", ""),
    ("83010203", [1, 2, 3]), ("a0", {}), ("a26161016162820203", {"a": 1, "b": [2, 3]}),
    ("f93e00", 1.5), ("fa47c35000", 100000.0), ("fb3ff199999999999a", 1.1),
    ("5f42010243030405ff", b"\x01\x02\x03\x04\x05"),
    ("7f657374726561646d696e67ff", "streaming"),
    ("9f018202039f0405ffff", [1, [2, 3], [4, 5]]),
    ("bf61610161629f0203ffff", {"a": 1, "b": [2, 3]}),
)


def self_test(fixtures_dir=None):
    failures = []
    for text, expected in CBOR_VECTORS:
        try:
            got = cbor_decode(bytes.fromhex(text))
        except CborError as error:
            failures.append("cbor %s: %s" % (text, error))
            continue
        if _canonical(got) != _canonical(expected):
            failures.append("cbor %s: got %r, expected %r" % (text, got, expected))
    for text in ("a2616101616101", "8301", "0000", "1c"):
        try:
            cbor_decode(bytes.fromhex(text))
            failures.append("cbor %s: malformed input was accepted" % text)
        except CborError:
            pass
    for length, expected in KNOWN_BLAKE3:
        if blake3_hex(_pattern(length)) != expected:
            failures.append("blake3 length %d does not match the published vector" % length)
    vectors = 0
    if fixtures_dir is not None:
        path = os.path.join(fixtures_dir, "blake3_vectors.json")
        if os.path.exists(path):
            with open(path, "r") as handle:
                listed = json.load(handle)
            for entry in listed["vectors"]:
                vectors += 1
                if blake3_hex(_pattern(entry["len"])) != entry["hash"]:
                    failures.append("blake3 length %d does not match vector file" % entry["len"])
    writer = Writer("t").word(1).text("ab")
    expected_stream = (struct.pack("<Q", 1) + b"t" + struct.pack("<Q", 1)
                       + struct.pack("<Q", 2) + b"ab")
    if bytes(writer.buf) != expected_stream:
        failures.append("canonical writer framing differs from the specification")
    return failures, vectors


def run_manifest(spec_dir, fixtures_dir):
    manifest_path = os.path.join(fixtures_dir, "manifest.json")
    with open(manifest_path, "r") as handle:
        manifest = json.load(handle)
    failures, vectors = self_test(fixtures_dir)
    if failures:
        for failure in failures:
            print("SELF-TEST FAIL %s" % failure)
        return 1
    if vectors == 0:
        print("SELF-TEST FAIL no BLAKE3 vector file was found beside the manifest")
        return 1
    print("self-test ok (%d published BLAKE3 vectors)" % (vectors + len(KNOWN_BLAKE3)))
    problems = 0
    for entry in manifest["fixtures"]:
        path = os.path.join(fixtures_dir, entry["file"])
        with open(path, "rb") as handle:
            data = handle.read()
        outcome = None
        try:
            result = validate_bytes(data, entry["kind"], spec_dir)
            outcome = ("accept", result)
        except Reject as rejection:
            outcome = ("reject", rejection)
        expected = entry["expect"]
        if outcome[0] != expected:
            problems += 1
            detail = outcome[1] if outcome[0] == "reject" else "accepted"
            print("FAIL %s: expected %s, got %s (%s)" % (entry["file"], expected, outcome[0], detail))
            continue
        if expected == "reject" and outcome[1].klass != entry["reject_class"]:
            problems += 1
            print("FAIL %s: rejected as %s, expected %s (%s)"
                  % (entry["file"], outcome[1].klass, entry["reject_class"], outcome[1].message))
            continue
        if expected == "accept":
            if outcome[1]["identity"] != entry["identity"]:
                problems += 1
                print("FAIL %s: identity %s differs from the manifest %s"
                      % (entry["file"], outcome[1]["identity"], entry["identity"]))
                continue
            meta_file = entry.get("meta_json")
            if meta_file is not None:
                with open(os.path.join(fixtures_dir, meta_file), "r") as handle:
                    listed = json.load(handle)
                if _canonical(listed) != _canonical(outcome[1]["meta"]):
                    problems += 1
                    print("FAIL %s: the JSON rendering differs from the decoded CBOR" % entry["file"])
                    continue
        print("ok   %s (%s)" % (entry["file"], expected))
    return 1 if problems else 0


def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    spec_dir = os.path.dirname(here)
    fixtures_dir = os.path.join(spec_dir, "conformance")
    kind = None
    retained = None
    files = []
    mode = "manifest"
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg in ("--spec-dir", "--fixtures", "--kind", "--retained-identity"):
            if index + 1 >= len(argv):
                print("missing value for %s" % arg, file=sys.stderr)
                return 2
            value = argv[index + 1]
            index += 1
            if arg == "--spec-dir":
                spec_dir = value
            elif arg == "--fixtures":
                fixtures_dir = value
            elif arg == "--kind":
                kind = value
                mode = "single"
            else:
                retained = value
        elif arg == "--self-test":
            mode = "self"
        elif arg == "--dump-tables":
            mode = "tables"
        elif arg in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            files.append(arg)
        index += 1
    if mode == "tables":
        print(json.dumps({
            "operation_tags": OP_TAG,
            "meaning_paradigms": MEANING_PARADIGM,
            "kind_stages": KIND_STAGE,
            "role_tags": ROLE_TAG,
            "calibration_tags": CALIBRATION_TAG,
        }, sort_keys=True))
        return 0
    if mode == "self":
        failures, vectors = self_test(fixtures_dir if os.path.isdir(fixtures_dir) else None)
        for failure in failures:
            print("FAIL %s" % failure)
        if not failures:
            print("self-test ok (%d vector-file entries)" % vectors)
        return 1 if failures else 0
    if mode == "single":
        if len(files) != 1:
            print("--kind needs exactly one file", file=sys.stderr)
            return 2
        with open(files[0], "rb") as handle:
            data = handle.read()
        failures, _ = self_test(None)
        if failures:
            print("SELF-TEST FAIL " + "; ".join(failures))
            return 1
        try:
            result = validate_bytes(data, kind, spec_dir, retained)
        except Reject as rejection:
            print("REJECT %s: %s" % (rejection.klass, rejection.message))
            return 1
        print("ACCEPT %s identity=%s" % (kind, result["identity"]))
        return 0
    try:
        return run_manifest(spec_dir, fixtures_dir)
    except (IOError, OSError, ValueError, KeyError) as error:
        print("cannot run the conformance manifest: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
