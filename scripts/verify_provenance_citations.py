#!/usr/bin/env python3
"""Resolve DOI titles through Crossref/DataCite; ordinary checks are offline.

Bibliographic identity is distinct from checking the cited theorem/algorithm or
its implementation. This tool never marks those scientific reviews complete.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tomllib
from _repo import REPO_ROOT

CACHE = "provenance/citation_metadata.json"


def normalize(title: str) -> str:
    plain = html.unescape(re.sub(r"<[^>]+>", "", title))
    return "".join(
        c for c in unicodedata.normalize("NFKC", plain).casefold() if c.isalnum()
    )


def citations(root: Path) -> list[dict]:
    result = []
    for path in sorted(root.glob("provenance/*.toml")):
        if path.name == "_template.toml":
            continue
        record = tomllib.loads(path.read_text(encoding="utf-8"))
        for index, paper in enumerate(record.get("papers", [])):
            result.append(
                {
                    "record": path.relative_to(root).as_posix(),
                    "paper": index,
                    "title": paper["title"],
                    "doi": paper.get("doi"),
                    "sections": paper.get("sections", []),
                }
            )
    return result


def registry_url(doi: str) -> str:
    base = (
        "https://api.datacite.org/dois/"
        if doi.lower().startswith("10.48550/")
        else "https://api.crossref.org/works/"
    )
    return base + urllib.parse.quote(doi, safe="/")


def fetch(doi: str) -> dict:
    # Public registries rate-limit bursts; regeneration is an occasional audit.
    time.sleep(1)
    url = registry_url(doi)
    request = urllib.request.Request(
        url, headers={"User-Agent": "Antecedent-provenance-audit/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read()
        data = json.loads(raw)
        if doi.startswith("10.48550/"):
            attrs = data["data"]["attributes"]
            titles = [t["title"] for t in attrs["titles"]]
            returned_doi = attrs["doi"]
        else:
            attrs = data["message"]
            titles = attrs["title"]
            returned_doi = attrs["DOI"]
        return {
            "url": url,
            "doi": returned_doi,
            "titles": titles,
            "response_sha256": hashlib.sha256(raw).hexdigest(),
        }
    except (OSError, ValueError, KeyError) as error:
        return {"url": url, "error": str(error)}


def verdict(citation: dict, cache: dict) -> str:
    doi = citation.get("doi")
    if not doi:
        return "no_doi_primary_source_review_required"
    item = cache.get(doi.lower())
    if not item:
        return "not_resolved"
    if item.get("error"):
        return "resolution_failed"
    if item.get("doi", "").lower() != doi.lower():
        return "doi_mismatch"
    if normalize(citation["title"]) not in {
        normalize(t) for t in item.get("titles", [])
    }:
        return "title_mismatch"
    return "bibliographic_identity_verified"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="network operation; retain primary metadata",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail on unresolved or mismatched DOI titles",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="also replace successful cached metadata"
    )
    parser.add_argument("--cache", type=Path, default=REPO_ROOT / CACHE)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="repository to inspect (for gate fault controls)",
    )
    args = parser.parse_args()
    rows = citations(args.root)
    cache = (
        json.loads(args.cache.read_text(encoding="utf-8"))
        if args.cache.is_file()
        else {}
    )
    if args.fetch:
        dois = sorted(
            {
                r["doi"].lower()
                for r in rows
                if r.get("doi")
                and (
                    args.refresh
                    or not cache.get(r["doi"].lower())
                    or cache[r["doi"].lower()].get("error")
                )
            }
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            futures = {pool.submit(fetch, doi): doi for doi in dois}
            for future in as_completed(futures):
                doi = futures[future]
                cache[doi] = future.result()
                print(
                    f"resolved {doi}: {'error' if cache[doi].get('error') else 'metadata'}",
                    flush=True,
                )
        args.cache.write_text(
            json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    failures = []
    for row in rows:
        status = verdict(row, cache)
        if row.get("doi") and status != "bibliographic_identity_verified":
            failures.append(
                f"{row['record']} papers[{row['paper']}]: {status}: {row['doi']}"
            )
    for failure in failures:
        print(f"FINDING: {failure}")
    print(
        f"DOI citations: {sum(bool(r.get('doi')) for r in rows)}; unresolved/mismatched: {len(failures)}. "
        "Theorem/algorithm sections and implementation correspondence require separate review."
    )
    return int(args.check and bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
