"""Cite the methods behind a result: sources, caveats, and the provenance records they come from.

Every executed route names the provenance records behind it: the identifier, the estimator and
any capability. :func:`provenance_of` (and ``result.method_provenance``) returns them, each with
its published sources, the deviations from the published method its record documents, what its
bounded method review leaves out, and the record id and digest. :func:`bibliography` collects
the sources of several results into one de-duplicated list, each source with the caveats of
every record that cites it, ready to paste into a methods section.

The table and the de-duplication are the Rust implementation's
(``antecedent_io::method_provenance``), generated from ``provenance/*.toml``. Provenance is
descriptive: it changes no number, support status or calibration annotation. See
``docs/guides/citing-a-study.md``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from . import _native

__all__ = [
    "Caveat",
    "Citation",
    "MethodProvenance",
    "ProvenanceRecord",
    "Source",
    "bibliography",
    "capability_provenance",
    "method_provenance",
    "provenance_of",
]


@dataclass(frozen=True)
class Source:
    """One published source: title, authors and year from the DOI registry, DOI or URL."""

    title: str
    authors: tuple[str, ...] = ()
    year: int | None = None
    doi: str | None = None
    #: Whether the DOI resolved to the recorded title in the retained registry metadata.
    doi_verified: bool = False
    url: str | None = None
    #: The algorithms, theorems or sections the implementation follows.
    sections: tuple[str, ...] = ()

    @classmethod
    def _from(cls, raw: dict[str, Any]) -> Source:
        return cls(
            title=raw["title"],
            authors=tuple(raw.get("authors", ())),
            year=raw.get("year"),
            doi=raw.get("doi"),
            doi_verified=bool(raw.get("doi_verified", False)),
            url=raw.get("url"),
            sections=tuple(raw.get("sections", ())),
        )


@dataclass(frozen=True)
class ProvenanceRecord:
    """One provenance record behind a route, reduced to what a citation needs."""

    #: Record id (``provenance/<id>.toml``).
    id: str
    #: SHA-256 of the record without its review tables.
    digest: str
    #: ``"method_review"`` or ``"non_algorithm_review"`` when that review attests this digest.
    review: str
    #: ``(role, key)`` pairs: ``("estimator", "aipw")``, ``("identifier", ...)``, ...
    cited_for: tuple[tuple[str, str], ...]
    sources: tuple[Source, ...]
    #: Documented departures from the published method.
    deviations: tuple[str, ...]
    #: What the bounded method review does not cover.
    scope_limits: tuple[str, ...]
    #: Records a wrapper record delegates its numerics to.
    covered_by: tuple[str, ...] = ()

    @classmethod
    def _from(cls, raw: dict[str, Any]) -> ProvenanceRecord:
        return cls(
            id=raw["id"],
            digest=raw["digest"],
            review=raw["review"],
            cited_for=tuple((c["role"], c["key"]) for c in raw.get("cited_for", ())),
            sources=tuple(Source._from(s) for s in raw.get("sources", ())),
            deviations=tuple(raw.get("deviations", ())),
            scope_limits=tuple(raw.get("scope_limits", ())),
            covered_by=tuple(raw.get("covered_by", ())),
        )


@dataclass(frozen=True)
class MethodProvenance:
    """Every provenance record behind one result's route."""

    records: tuple[ProvenanceRecord, ...]
    #: ``(role, key)`` route components that name no provenance record.
    unlinked: tuple[tuple[str, str], ...]
    _json: str

    @classmethod
    def _parse(cls, text: str) -> MethodProvenance:
        raw = json.loads(text)
        return cls(
            records=tuple(ProvenanceRecord._from(r) for r in raw.get("records", ())),
            unlinked=tuple((c["role"], c["key"]) for c in raw.get("unlinked", ())),
            _json=text,
        )

    def record(self, record_id: str) -> ProvenanceRecord | None:
        """The cited record with ``record_id``."""
        return next((r for r in self.records if r.id == record_id), None)

    def citations(self) -> tuple[Citation, ...]:
        """This result's sources, de-duplicated, each with its records' caveats."""
        return _bibliography([self])

    def to_dict(self) -> dict[str, Any]:
        """The provenance as the JSON-ready mapping the artifact section carries."""
        return json.loads(self._json)


@dataclass(frozen=True)
class Caveat:
    """What was not implemented as published, stated beside a citation."""

    #: Record that documents it.
    record: str
    #: ``"deviation"`` or ``"scope_limit"``.
    kind: str
    text: str


@dataclass(frozen=True)
class Citation:
    """One bibliography entry: a source, the records citing it, and their caveats."""

    source: Source
    #: ``(record id, digest)`` of every record citing the source.
    records: tuple[tuple[str, str], ...]
    caveats: tuple[Caveat, ...]
    #: One-line reference: ``Authors (year). Title. https://doi.org/<doi>``.
    reference: str

    def __str__(self) -> str:
        return self.reference


def method_provenance(
    identifier: str | None = None, estimator: str | None = None
) -> MethodProvenance:
    """Provenance of the route with these strategy-table identifier and estimator ids."""
    return MethodProvenance._parse(_native.method_provenance_json(identifier, estimator))


def capability_provenance(capability: str) -> MethodProvenance:
    """Provenance of a capability-specific route, by capability id."""
    return MethodProvenance._parse(_native.capability_provenance_json(capability))


def provenance_of(result: Any) -> MethodProvenance:
    """Provenance of ``result``: the identifier and estimator its executed plan names.

    ``result`` is an :class:`~antecedent.outcomes.Outcome` or the result it holds. A
    :class:`MethodProvenance` passes through. A result without a plan (a refusal, or a family
    that records no strategy-table route) names no route and has empty provenance.
    """
    from .outcomes import Outcome

    if isinstance(result, MethodProvenance):
        return result
    if isinstance(result, Outcome):
        result = result.value
    plan = getattr(result, "plan", None)
    identifier = getattr(plan, "identifier", None) if plan is not None else None
    estimator = getattr(plan, "estimator", None) if plan is not None else None
    return method_provenance(identifier or None, estimator or None)


def _bibliography(provenances: Iterable[MethodProvenance]) -> tuple[Citation, ...]:
    raw = json.loads(_native.bibliography_json([p._json for p in provenances]))
    return tuple(
        Citation(
            source=Source._from(entry["source"]),
            records=tuple((rid, digest) for rid, digest in entry["records"]),
            caveats=tuple(
                Caveat(record=c["record"], kind=c["kind"], text=c["text"]) for c in entry["caveats"]
            ),
            reference=entry["reference"],
        )
        for entry in raw
    )


def bibliography(*results: Any) -> tuple[Citation, ...]:
    """The bibliography of a study: the sources of every result, de-duplicated across results.

    Each argument is a result, a :class:`MethodProvenance`, or a list or tuple of them. A source
    cited by several results appears once, with every record that cites it and every caveat
    those records document.
    """
    flat: list[Any] = []
    for item in results:
        # Result models iterate over their fields, so only plain sequences are flattened.
        if isinstance(item, (list, tuple)):
            flat.extend(item)
        else:
            flat.append(item)
    return _bibliography(provenance_of(item) for item in flat)
