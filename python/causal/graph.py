"""Graph helpers (DESIGN.md §25.1 DOT+JSON interchange)."""

from __future__ import annotations

from ._native import format_dag_dot, format_dag_json, parse_dag_dot, parse_dag_json

__all__ = [
    "format_dag_dot",
    "format_dag_json",
    "parse_dag_dot",
    "parse_dag_json",
]
