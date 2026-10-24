#!/usr/bin/env python3
"""Run a repository example as a timed control, in either build.

The example under `--example` (a path relative to the working directory, which the lane
runner sets to the repository for the candidate and to the frozen v2.4.1 tree for the
control) is executed whole inside the timed section, so its requested uncertainty and
diagnostics are in the timing and the native resource measurement. Its printed output is
captured and digested; the text itself is retained under `_printed`.

    python parity/performance_lanes/entry_points/run_example.py --role control \\
        --example examples/python/propensity_weighting.py
"""

from __future__ import annotations

import contextlib
import io
import runpy
import sys
from pathlib import Path
from typing import Any

from _entry import arguments, digest_fields, run_entry


def main() -> None:
    args = arguments(
        __doc__ or "",
        lambda parser: parser.add_argument("--example", required=True, help="example script"),
    )
    example = Path(args.example).resolve()
    if not example.is_file():
        sys.exit(f"example {args.example} does not exist under {Path.cwd()}")
    sys.path.insert(0, str(example.parent))
    name = example.stem

    def section() -> Any:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            runpy.run_path(str(example), run_name="__main__")
        return buffer.getvalue()

    def fields(printed: str) -> dict[str, Any]:
        lines = [line for line in printed.splitlines() if line.strip()]
        return {"printed_lines": len(lines), "_printed": lines}

    run_entry(
        f"example:{name}",
        args,
        digest_fields({"example": args.example}),
        section,
        fields,
        required=("_printed",),
    )


if __name__ == "__main__":
    main()
