#!/usr/bin/env python3
"""Generate the versioned host-integration JSON Schema.

The schema is not written by hand. Every enumeration in it is read from the
sources that own it:

* the reason-code vocabulary from ``parity/reason_codes.toml``;
* claim kinds, domain statuses and the capability vocabularies from the Rust
  enums that spell them (``named_variants!`` blocks and ``as_str`` matches);
* the support-matrix query, graph-class and structure-source labels from the
  facade's support module;
* the exception families from the native exception declarations and
  ``python/antecedent/errors.py``;
* the durable artifact format from ``STABLE_FORMAT``.

The Rust test ``host_contract`` re-checks the committed file against the
compiled types, and ``python/tests/test_host_contract.py`` re-checks it against
the installed package, so a drift in either direction fails a test.

    python3 scripts/generate_host_schema.py          # rewrite the committed file
    python3 scripts/generate_host_schema.py --check  # fail when it is stale
"""

from __future__ import annotations

import json
import re
import sys

from _repo import REPO_ROOT as ROOT
from _repo import load_toml, read_text, write_text

OUT = ROOT / "crates" / "antecedent" / "schemas" / "host_contract.schema.json"

#: Bumped when a consumer of the schema must change (a removed or renamed
#: member, or a meaning that changed). New enumeration members and new
#: optional fields do not change it.
CONTRACT_VERSION = 1

ARM = re.compile(r'Self::\w+(?:\s*\{[^}]*\})?\s*=>\s*"([^"]+)"')


def _slice_after(text: str, marker: str, path: str) -> str:
    start = text.find(marker)
    if start < 0:
        raise SystemExit(f"{path}: cannot find `{marker}`")
    return text[start:]


def named_variants(rel: str, type_name: str) -> list[str]:
    """Names listed by one ``named_variants!`` block, in declaration order."""
    text = read_text(ROOT / rel)
    block = _slice_after(text, f"    {type_name} {{\n", rel)
    block = block[: block.index("\n    }\n")]
    names = re.findall(r'=>\s*"([^"]+)"', block)
    if not names:
        raise SystemExit(f"{rel}: no variants for {type_name}")
    return names


def as_str_names(rel: str, impl_name: str) -> list[str]:
    """Names returned by ``impl <impl_name>``'s first ``as_str`` match."""
    text = read_text(ROOT / rel)
    block = _slice_after(text, f"impl {impl_name} {{", rel)
    block = _slice_after(block, "pub const fn as_str", rel)
    block = _slice_after(block, "match self {", rel)
    block = block[: block.index("\n        }\n")]
    names = ARM.findall(block)
    if not names:
        raise SystemExit(f"{rel}: no names for {impl_name}")
    return names


def reason_codes() -> tuple[list[str], list[str]]:
    rows = load_toml(ROOT / "parity" / "reason_codes.toml")["code"]
    every = sorted({row["id"] for row in rows})
    runtime = sorted(
        {row["id"] for row in rows if "runtime_refusal" in row.get("applies_to", [])}
    )
    return every, runtime


def snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def family_code(class_name: str) -> str:
    """Stable family code of an exception class name."""
    special = {
        "CausalError": "error",
        "EffectNotIdentified": "not_identified",
        "ReviewRequired": "review_required",
        "StructuredRefusal": "structured_refusal",
        "CausalReviewError": "review_required",
        "CausalUnsupportedError": "unsupported",
        "CausalCancelledError": "cancelled",
    }
    if class_name in special:
        return special[class_name]
    core = class_name.removeprefix("Causal").removesuffix("Error")
    return snake(core)


def error_families() -> list[dict[str, object]]:
    """Native families from their declarations, then the Python-only narrowings."""
    declarations = read_text(ROOT / "python" / "src" / "errors.rs")
    native = re.findall(
        r"create_exception!\(antecedent\._native,\s*(\w+),\s*(\w+)\)", declarations
    )
    if not native:
        raise SystemExit("python/src/errors.rs: no native exception declarations")
    families: list[dict[str, object]] = []
    for name, parent in native:
        families.append(
            {
                "code": family_code(name),
                "python_class": name,
                "parent": None if name == "CausalError" else family_code(parent),
                "origin": "native",
                "narrows": None,
            }
        )
    errors = read_text(ROOT / "python" / "antecedent" / "errors.py")
    for name, bases in re.findall(r"^class (\w+)\(([^)]*)\):", errors, re.M):
        if name in {"CausalUnsupportedError", "ReviewRequired"}:
            continue  # redefined over a native class; the native entry owns the code
        first = next(
            (b.strip() for b in bases.split(",") if b.strip().startswith(("Causal", "Structured"))),
            None,
        )
        if first is None:
            continue
        parent = "unsupported" if first == "CausalUnsupportedError" else family_code(first)
        code = family_code(name)
        families.append(
            {
                "code": code,
                "python_class": name,
                "parent": parent,
                "origin": "python",
                # Rust reports the parent for a failure Python narrows; not_identified is a
                # family in its own right on both sides.
                "narrows": None if code == "not_identified" else parent,
            }
        )
    codes = [f["code"] for f in families]
    if len(set(codes)) != len(codes):
        raise SystemExit(f"duplicate error family codes: {codes}")
    return families


def artifact_format() -> dict[str, int]:
    text = read_text(ROOT / "crates" / "antecedent-io-core" / "src" / "migrate.rs")
    match = re.search(
        r"STABLE_FORMAT: FormatVersion = FormatVersion \{ major: (\d+), minor: (\d+) \}", text
    )
    if match is None:
        raise SystemExit("cannot read STABLE_FORMAT")
    return {"major": int(match.group(1)), "minor": int(match.group(2))}


def enum_def(values: list[str], description: str) -> dict[str, object]:
    return {"type": "string", "enum": values, "description": description}


def build() -> dict[str, object]:
    every_code, refusal_codes = reason_codes()
    families = error_families()
    hexid = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    nullable_hex = {"oneOf": [{"$ref": "#/$defs/digest"}, {"type": "null"}]}
    claim = "crates/antecedent-contract/src/claim.rs"
    capability = "crates/antecedent-contract/src/capability.rs"
    cell = "crates/antecedent/src/support/cell.rs"
    defs: dict[str, object] = {
        "digest": {**hexid, "description": "Lowercase hex of a 32-byte identity digest."},
        "host_status": enum_def(
            ["supported", "partial", "unsupported"],
            "What a host may do with a claim or an operation. supported: use it as stated. "
            "partial: use only what the record says (an identified set, an open obligation, an "
            "unreviewed graph, an unknown domain). unsupported: do not use it; read the refusal.",
        ),
        "refusal_code": enum_def(
            refusal_codes,
            "Registered runtime-refusal reason code. A refusal never carries another code.",
        ),
        "reason_code": enum_def(
            every_code,
            "Every registered reason code, including the calibration and product vocabularies.",
        ),
        "error_family": enum_def(
            [str(f["code"]) for f in families],
            "Stable exception family. The same code names the Rust CausalError class and the "
            "Python exception class.",
        ),
        "claim_kind": enum_def(
            named_variants(claim, "ClaimKind"), "Kind of value a claim carries."
        ),
        "domain_status": enum_def(
            named_variants(claim, "DomainStatus"),
            "Status of one claim domain (identification, support or evaluated).",
        ),
        "semantic_applicability": enum_def(
            as_str_names(capability, "SemanticApplicability"),
            "Support-matrix applicability of a requested operation.",
        ),
        "operation_readiness": enum_def(
            as_str_names(capability, "OperationReadiness"),
            "Snapshot readiness of an operation. Only executable means ready to run.",
        ),
        "operation_kind": enum_def(
            as_str_names(capability, "OperationKind"), "Operation a capability report describes."
        ),
        "support_cell_status": enum_def(
            as_str_names(cell, "CellStatus"), "Classification of one support-matrix cell."
        ),
        "support_query": enum_def(
            as_str_names(cell, "SupportQuery"), "Public support-matrix query labels."
        ),
        "graph_class": enum_def(
            as_str_names("crates/antecedent/src/accepted.rs", "GraphClass"),
            "Graph class of the accepted structure.",
        ),
        "structure_source": enum_def(
            as_str_names(cell, "StructureSource"), "How the structure was supplied."
        ),
        "inference": enum_def(["frequentist", "bayesian"], "Inference family of a program."),
        "refusal": {
            "type": "object",
            "description": "A typed refusal. It names the registered code and, when known, what "
            "the caller can change to proceed. It is never a result.",
            "required": ["status", "family", "reason_code", "message"],
            "properties": {
                "status": {"const": "unsupported"},
                "family": {"$ref": "#/$defs/error_family"},
                "reason_code": {
                    "oneOf": [{"$ref": "#/$defs/refusal_code"}, {"type": "null"}],
                    "description": "Null only for a failure that is not a refusal (a malformed "
                    "input, a resource limit, a cancellation); family still classifies it.",
                },
                "message": {"type": "string"},
                "remedy": {"type": ["string", "null"]},
            },
            "additionalProperties": True,
        },
        "claim_domains": {
            "type": "object",
            "required": ["identification", "support", "evaluated"],
            "properties": {
                "identification": {"$ref": "#/$defs/domain_status"},
                "support": {"$ref": "#/$defs/domain_status"},
                "evaluated": {"$ref": "#/$defs/domain_status"},
            },
            "additionalProperties": False,
        },
        "claim_envelope": {
            "type": "object",
            "description": "Portable claim over a compiled contract and one execution. A claim of "
            "kind refusal or incomplete is a record, not a value.",
            "required": ["claim_id", "kind", "domains"],
            "properties": {
                "claim_id": {"$ref": "#/$defs/digest"},
                "kind": {"$ref": "#/$defs/claim_kind"},
                "causal": {
                    "type": "boolean",
                    "description": "False for an associational claim: identification is not "
                    "claimed, so the claim is never an intervention effect and is not counted "
                    "as a causal one.",
                },
                "value": {"type": ["number", "null"]},
                "outcome_units": {"type": ["string", "null"]},
                "domains": {"$ref": "#/$defs/claim_domains"},
                "execution": nullable_hex,
                "evidence": {"type": "array", "items": {"type": "string"}},
                "status": {"$ref": "#/$defs/host_status"},
            },
            "additionalProperties": True,
        },
        "contract_identities": {
            "type": "object",
            "description": "Domain-separated identities of a compiled contract. program is null "
            "before preparation: a structural inspection has not identified anything.",
            "required": ["target", "identification", "inference_binding", "observation"],
            "properties": {
                "target": {"$ref": "#/$defs/digest"},
                "identification": {"$ref": "#/$defs/digest"},
                "identification_product": nullable_hex,
                "program": nullable_hex,
                "inference_binding": {"$ref": "#/$defs/digest"},
                "observation": {"$ref": "#/$defs/digest"},
                "data_snapshot": {"$ref": "#/$defs/digest"},
                "execution": nullable_hex,
            },
            "additionalProperties": True,
        },
        "causal_contract": {
            "type": "object",
            "description": "Immutable record of what was compiled: identities, structure, "
            "support status and the four reasoning slots (identification, support, uncertainty, "
            "assumptions). A slot is either available or states why it is not.",
            "required": ["identities", "graph_class", "structure_source", "query_kind"],
            "properties": {
                "identities": {"$ref": "#/$defs/contract_identities"},
                "graph_class": {"$ref": "#/$defs/graph_class"},
                "structure_source": {"$ref": "#/$defs/structure_source"},
                "support_status": {
                    "oneOf": [{"$ref": "#/$defs/support_cell_status"}, {"type": "null"}]
                },
                "query_kind": {"type": "string"},
                "inference": {"$ref": "#/$defs/inference"},
                "identifier": {"type": ["string", "null"]},
                "estimator": {"type": ["string", "null"]},
                "row_count": {"type": "integer", "minimum": 0},
                "modality": {"type": "string"},
                "reasoning": {"$ref": "#/$defs/reasoning"},
            },
            "additionalProperties": True,
        },
        "reasoning_slot": {
            "type": "object",
            "required": ["available", "summary"],
            "properties": {
                "available": {"type": "boolean"},
                "reason": {"type": ["string", "null"]},
                "summary": {"type": "string"},
                "payload": {"type": "object"},
            },
            "additionalProperties": True,
        },
        "reasoning": {
            "type": "object",
            "required": ["identification", "support", "uncertainty", "assumptions"],
            "properties": {
                slot: {"$ref": "#/$defs/reasoning_slot"}
                for slot in ("identification", "support", "uncertainty", "assumptions")
            },
            "additionalProperties": True,
        },
        "blocker": {
            "type": "object",
            "required": ["id", "reason", "scientific"],
            "properties": {
                "id": {"type": "string"},
                "reason": {"type": "string"},
                "scientific": {
                    "type": "boolean",
                    "description": "False for a budget, resource or cancellation blocker: it "
                    "says nothing about whether the question can be answered.",
                },
            },
            "additionalProperties": False,
        },
        "next_action": {
            "type": "object",
            "required": ["id", "description"],
            "properties": {"id": {"type": "string"}, "description": {"type": "string"}},
            "additionalProperties": False,
        },
        "licensed_neighbor": {
            "type": "object",
            "description": "A licensed alternative that changes named premises. An explicit "
            "choice for the caller, never an automatic fallback.",
            "required": ["coordinate", "changed", "required_action"],
            "properties": {
                "coordinate": {"type": "string"},
                "changed": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["axis", "from", "to"],
                        "properties": {
                            "axis": {"type": "string"},
                            "from": {"type": "string"},
                            "to": {"type": "string"},
                        },
                        "additionalProperties": False,
                    },
                },
                "required_action": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "unmet_obligation": {
            "type": "object",
            "description": "An obligation of the contract that is not yet resolved: a check "
            "that has not run or has failed.",
            "required": ["id", "scope", "kind", "status", "description"],
            "properties": {
                "id": {"type": "string"},
                "scope": {"type": "string"},
                "kind": {"type": "string"},
                "status": {"type": "string"},
                "required_check": {"type": ["string", "null"]},
                "description": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "operation_report": {
            "type": "object",
            "description": "Capability report for one operation, read before running it.",
            "required": ["operation", "applicability", "blockers", "next_actions"],
            "properties": {
                "operation": {"$ref": "#/$defs/operation_kind"},
                "applicability": {"$ref": "#/$defs/semantic_applicability"},
                "readiness": {
                    "oneOf": [{"$ref": "#/$defs/operation_readiness"}, {"type": "null"}]
                },
                "coordinate": {"type": ["string", "null"]},
                "blockers": {"type": "array", "items": {"$ref": "#/$defs/blocker"}},
                "next_actions": {"type": "array", "items": {"$ref": "#/$defs/next_action"}},
                "neighbors": {"type": "array", "items": {"$ref": "#/$defs/licensed_neighbor"}},
                "unmet_obligations": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/unmet_obligation"},
                },
                "status": {"$ref": "#/$defs/host_status"},
            },
            "additionalProperties": True,
        },
        "fallback_unavailable": {
            "type": "object",
            "description": "A part of a descriptive fallback that is not provided, with the typed "
            "reason it is not. Absence is never silent.",
            "required": ["available", "part", "reason_code", "detail", "message"],
            "properties": {
                "available": {"const": False},
                "part": {"enum": ["associational", "evidence_plan"]},
                "reason_code": {"type": "string"},
                "detail": {"type": "string"},
                "message": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "evidence_obligation": {
            "type": "object",
            "description": "Evidence an identification failure owes, with the proof step that "
            "derives it. An establish_assumption obligation is satisfiable by no study.",
            "required": ["id", "kind", "scope", "variables", "reason", "satisfiable_by_study"],
            "properties": {
                "id": {"type": "string"},
                "kind": {"type": "string"},
                "scope": {"type": "string"},
                "variables": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
                "family": {"type": "string"},
                "proof_step": {"type": ["string", "null"]},
                "satisfiable_by_study": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        "fallback_response": {
            "type": "object",
            "description": "The opt-in descriptive fallback of an identification-class refusal. "
            "The causal question stays refused: status and causal_status are unsupported and "
            "refusal is the unchanged refusal. associational is a noncausal description of the "
            "observed law (claim.causal is false, it ranks no action, and it is not an answer to "
            "the causal question) or the typed reason there is none; evidence_plan carries the "
            "obligations, next actions and ranked plan or the typed reason there is none. "
            "associational_status shows the status of the noncausal claim beside the causal one.",
            "required": [
                "status",
                "causal_status",
                "associational_status",
                "refusal",
                "associational",
                "evidence_plan",
                "notice",
            ],
            "properties": {
                "status": {"const": "unsupported"},
                "causal_status": {"const": "unsupported"},
                "associational_status": {
                    "oneOf": [{"$ref": "#/$defs/host_status"}, {"const": "unavailable"}]
                },
                "refusal": {"$ref": "#/$defs/refusal"},
                "associational": {
                    "oneOf": [
                        {
                            "type": "object",
                            "required": ["available", "noncausal", "ranks_actions", "claim"],
                            "properties": {
                                "available": {"const": True},
                                "noncausal": {"const": True},
                                "ranks_actions": {"const": False},
                                "rule": {"type": "string"},
                                "claim": {"$ref": "#/$defs/claim_envelope"},
                            },
                            "additionalProperties": False,
                        },
                        {"$ref": "#/$defs/fallback_unavailable"},
                    ]
                },
                "evidence_plan": {
                    "oneOf": [
                        {
                            "type": "object",
                            "required": ["available", "obligations", "next_actions", "ranked_plan"],
                            "properties": {
                                "available": {"const": True},
                                "obligations": {
                                    "type": "array",
                                    "items": {"$ref": "#/$defs/evidence_obligation"},
                                },
                                "next_actions": {
                                    "type": "array",
                                    "items": {"$ref": "#/$defs/next_action"},
                                },
                                "ranked_plan": {"type": "object"},
                            },
                            "additionalProperties": False,
                        },
                        {"$ref": "#/$defs/fallback_unavailable"},
                    ]
                },
                "notice": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "analysis_request": {
            "type": "object",
            "description": "What a host submits to compile an analysis. The query is named by its "
            "support-matrix label; its remaining members are that query's own parameters. Data "
            "stay in the host: a request carries column names or a data reference, not a callback.",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "object",
                    "required": ["kind"],
                    "properties": {
                        "kind": {
                            "description": "A support-matrix query label, or the name of a "
                            "route-specific query the host's version of the library documents.",
                            "anyOf": [
                                {"$ref": "#/$defs/support_query"},
                                {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9]*$"},
                            ],
                        }
                    },
                    "additionalProperties": True,
                },
                "graph": {
                    "type": "object",
                    "properties": {
                        "edges": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                        }
                    },
                    "additionalProperties": True,
                },
                "data": {
                    "type": "object",
                    "properties": {
                        "columns": {"type": "array", "items": {"type": "string"}},
                        "reference": {"type": "string"},
                    },
                    "additionalProperties": True,
                },
                "options": {
                    "type": "object",
                    "properties": {
                        "inference": {"$ref": "#/$defs/inference"},
                        "seed": {"type": "integer", "minimum": 0},
                        "threads": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "0 lets the library choose; a positive value pins the "
                            "study to that many worker threads.",
                        },
                        "bootstrap": {"type": "integer", "minimum": 0},
                    },
                    "additionalProperties": True,
                },
            },
            "additionalProperties": False,
        },
        "host_response": {
            "description": "What a host reads back for one request. Exactly one of claim or "
            "refusal is present. A claim of kind refusal, or one whose domain is contradicted, "
            "is carried as a claim with status unsupported.",
            "oneOf": [
                {
                    "type": "object",
                    "required": ["status", "claim"],
                    "properties": {
                        "status": {"$ref": "#/$defs/host_status"},
                        "claim": {"$ref": "#/$defs/claim_envelope"},
                        "contract": {"$ref": "#/$defs/causal_contract"},
                    },
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "required": ["status", "refusal"],
                    "properties": {
                        "status": {"const": "unsupported"},
                        "refusal": {"$ref": "#/$defs/refusal"},
                    },
                    "additionalProperties": False,
                },
            ],
        },
        "artifact_expectation": {
            "type": "object",
            "description": "What a consumer retains independently of the artifact bytes and "
            "supplies when loading. Container metadata never supplies it.",
            "properties": {
                "kind": {"type": "string"},
                "identities": {"type": "object"},
            },
            "additionalProperties": True,
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"urn:antecedent:host-contract:{CONTRACT_VERSION}",
        "title": "Antecedent host-integration contract",
        "description": "Machine-readable vocabulary an application or tool layer uses to tell a "
        "supported claim from a partial answer or a refusal before using it. Generated by "
        "scripts/generate_host_schema.py; do not edit.",
        "x-contract-version": CONTRACT_VERSION,
        "x-artifact-format": artifact_format(),
        "x-error-families": families,
        "x-status-rules": {
            "claim": [
                "kind refusal: unsupported",
                "any domain contradicted or outside_scope: unsupported",
                "kind incomplete or bounds: partial",
                "kind associational, identification not_claimed and evaluated evaluated: "
                "supported (a noncausal claim)",
                "kind associational otherwise: partial",
                "identification identified and evaluated evaluated (support may be unknown): "
                "supported",
                "otherwise: partial",
            ],
            "operation": [
                "applicability impossible or unlicensed: unsupported",
                "readiness empirical_support_failed: unsupported",
                "applicability licensed and readiness executable: supported",
                "otherwise: partial",
            ],
        },
        "type": "object",
        "properties": {
            "request": {"$ref": "#/$defs/analysis_request"},
            "response": {"$ref": "#/$defs/host_response"},
            "contract": {"$ref": "#/$defs/causal_contract"},
            "claim": {"$ref": "#/$defs/claim_envelope"},
            "refusal": {"$ref": "#/$defs/refusal"},
            "operation_report": {"$ref": "#/$defs/operation_report"},
            "fallback": {"$ref": "#/$defs/fallback_response"},
            "status": {"$ref": "#/$defs/host_status"},
        },
        "$defs": defs,
    }


def render() -> str:
    return json.dumps(build(), indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    text = render()
    if "--check" in sys.argv[1:]:
        current = read_text(OUT) if OUT.exists() else ""
        if current != text:
            print(
                f"{OUT.relative_to(ROOT)} is stale; run python3 scripts/generate_host_schema.py",
                file=sys.stderr,
            )
            return 1
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_text(OUT, text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
