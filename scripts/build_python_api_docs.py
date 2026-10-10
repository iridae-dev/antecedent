#!/usr/bin/env python3
"""Build installed-wheel API documentation, including lazy public refusal classes."""

from __future__ import annotations

import argparse
from pathlib import Path


def build(output: Path) -> None:
    import antecedent
    import antecedent.errors as errors
    import pdoc

    # pdoc inspects module dictionaries. Resolve the declared lazy exports in
    # this docs process so each refusal has its class documentation and fields.
    lazy_names = [name for name in errors.__all__ if name not in vars(errors)]
    for name in errors.__all__:
        getattr(errors, name)
    module = pdoc.doc.Module.from_name("antecedent.errors")
    missing = [
        name for name in lazy_names if not isinstance(module.members.get(name), pdoc.doc.Class)
    ]
    if missing:
        raise RuntimeError(f"Public error classes missing from API docs: {missing}")

    print("antecedent", antecedent.__version__, antecedent.__file__)
    pdoc.pdoc("antecedent", output_directory=output)
    if not (output / "antecedent.html").is_file():
        raise RuntimeError("pdoc did not produce the package API page")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="directory for generated API HTML")
    args = parser.parse_args()
    build(args.output.resolve())


if __name__ == "__main__":
    main()
