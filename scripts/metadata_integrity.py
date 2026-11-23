"""Scoped metadata checks; these inspect declarations, not fixture prose."""
import re

ORCID = re.compile(r"(?:https?://orcid\.org/)?([0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{3}[0-9X])")
PACKAGE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")

def author_orcid_problems(citation, creators):
    match = re.search(r"(?ms)^authors:\s*\n(.*?)(?=^[^ #\t\n][^\n]*:|\Z)", citation)
    if match is None:
        return ["CITATION.cff has no authors section"]
    values = []
    for line in match.group(1).splitlines():
        m = re.match(r"^\s*(?:-\s*)?orcid:\s*(.*?)\s*(?:#.*)?$", line)
        if m:
            raw = m.group(1).strip().strip("\"'")
            if not ORCID.fullmatch(raw):
                return ["CITATION.cff author has invalid ORCID"]
            values.append(ORCID.fullmatch(raw).group(1))
    if not isinstance(creators, list):
        return [".zenodo.json creators must be an array"]
    zen = []
    for creator in creators:
        if not isinstance(creator, dict):
            return [".zenodo.json creator must be an object"]
        if "orcid" in creator:
            raw = creator["orcid"]
            found = ORCID.fullmatch(raw) if isinstance(raw, str) else None
            if found is None:
                return [".zenodo.json creator has invalid ORCID"]
            zen.append(found.group(1))
    if not values or not zen:
        return ["creator ORCIDs must be nonempty in both metadata files"]
    if len(values) != len(set(values)) or len(zen) != len(set(zen)):
        return ["creator ORCIDs must not be repeated"]
    if set(values) != set(zen):
        return ["creator ORCIDs differ between CITATION.cff and .zenodo.json"]
    return []

def oracle_rows(ledger):
    rows = ledger.get("method")
    if not isinstance(rows, list) or not rows or any(not isinstance(x, dict) for x in rows):
        return [], ["oracle_closure: method must be a nonempty array of tables"]
    return rows, []

def oracle_packages(fixture):
    if not isinstance(fixture, dict):
        return set()
    names = set()
    for key in ("oracle", "reference"):
        section = fixture.get(key)
        if not isinstance(section, dict):
            continue
        for field in ("project", "projects", "packages", "package", "versions"):
            value = section.get(field)
            pieces = ([value] if isinstance(value, str) else list(value) if isinstance(value, (list, dict)) else [])
            for piece in pieces:
                if isinstance(piece, str):
                    names.update(PACKAGE.findall(piece.lower()))
            # A projects table names separate typed implementations; inspect
            # only each nested project's explicit project declaration.
            if field == "projects" and isinstance(value, dict):
                for project in value.values():
                    declared = project.get("project") if isinstance(project, dict) else None
                    if isinstance(declared, str):
                        names.update(PACKAGE.findall(declared.lower()))
        # Explicit cross-check declaration is scoped to the oracle itself.
        cross_check = section.get("cross_check")
        if isinstance(cross_check, str):
            names.update(PACKAGE.findall(cross_check.lower()))
    return names
