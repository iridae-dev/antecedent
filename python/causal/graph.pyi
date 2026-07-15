"""Graph helpers (DESIGN.md §25.1 DOT+JSON interchange)."""

from ._native import (
    format_dag_dot as format_dag_dot,
    format_dag_json as format_dag_json,
    parse_dag_dot as parse_dag_dot,
    parse_dag_json as parse_dag_json,
)

__all__: list[str]
