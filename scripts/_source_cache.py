"""Dependency-aware memoization for synchronous source readers.

Cached closures inherit files read by nested source readers. Replacing an
assertion or a transitive helper must invalidate the cached proof in the same
process, including fault-injection tests. This is a cache, not an integrity
attestation; durable evidence uses content digests separately.
"""

from __future__ import annotations

import functools
from pathlib import Path

_active: list[dict] = []
_validation: dict = {}
# Bumped whenever the per-run stat memo is cleared, i.e. at the start of every
# top-level cached call. An entry validated in the current generation was checked
# against the same stat results, so nested lookups skip re-checking its dependencies.
_generation = [0]


def signature(path: Path):
    if path in _validation:
        return _validation[path]
    try:
        stat = path.stat()
    except FileNotFoundError:
        value = None
    else:
        value = stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino
    _validation[path] = value
    return value


def source_cache(function):
    entries = {}

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        if not _active:
            _validation.clear()
            _generation[0] += 1
        key = args, tuple(sorted(kwargs.items()))
        saved = entries.get(key)
        if saved is not None:
            result, dependencies, checked = saved
            if checked == _generation[0] or all(
                signature(path) == stamp for path, stamp in dependencies.items()
            ):
                entries[key] = result, dependencies, _generation[0]
                if _active:
                    _active[-1].update(dependencies)
                return result
        dependencies = {
            path: signature(path)
            for path in (*args, *kwargs.values())
            if isinstance(path, Path)
        }
        _active.append(dependencies)
        try:
            result = function(*args, **kwargs)
        finally:
            _active.pop()
            if _active:
                _active[-1].update(dependencies)
        entries[key] = result, dependencies, _generation[0]
        return result

    wrapped.cache_clear = entries.clear
    return wrapped
