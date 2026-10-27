#!/usr/bin/env python3
"""Execute actual notebook cells in isolated kernels and publish output previews."""

import argparse
import hashlib
import html as html_escape
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "target/doc-notebook-previews")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter with the matching Antecedent and notebook dependencies",
    )
    parser.add_argument("--timeout", type=int, default=600, help="seconds per cell")
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=180,
        help="seconds for an isolated kernel to become ready",
    )
    args = parser.parse_args()
    import nbformat
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter

    args.output.mkdir(parents=True, exist_ok=True)
    measurements = []
    release = subprocess.check_output(
        [args.python, "-c", "import antecedent; print(antecedent.__version__)"],
        text=True,
    ).strip()

    with TemporaryDirectory(prefix="antecedent-docs-kernel-") as directory:
        kernel = Path(directory) / "antecedent-docs"
        kernel.mkdir()
        kernel.joinpath("kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        str(Path(args.python).absolute()),
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": "Antecedent docs",
                    "language": "python",
                    "env": {
                        "MPLBACKEND": "Agg",
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "RAYON_NUM_THREADS": "1",
                    },
                }
            ),
            encoding="utf-8",
        )
        for path in sorted((ROOT / "examples/notebooks").glob("*.ipynb")):
            start = time.monotonic()
            notebook = nbformat.read(path, as_version=4)
            manager = KernelManager(
                kernel_name="antecedent-docs",
                kernel_spec_manager=KernelSpecManager(kernel_dirs=[directory]),
            )
            client = NotebookClient(
                notebook,
                timeout=args.timeout,
                startup_timeout=args.startup_timeout,
                km=manager,
                resources={"metadata": {"path": str(path.parent)}},
            )
            client.execute(cleanup_kc=True)
            outputs = sum(
                len(cell.get("outputs", [])) for cell in notebook.cells if cell.cell_type == "code"
            )
            if not outputs:
                raise RuntimeError(f"{path.name}: no executed output")
            elapsed = time.monotonic() - start
            html, _ = HTMLExporter(template_name="lab").from_notebook_node(notebook)
            html = re.sub(
                r'(<h[1-6]\b[^>]*\bid=")([^"]*)(")',
                lambda match: (
                    match[1] + html_escape.escape(unquote(match[2]), quote=True) + match[3]
                ),
                html,
            )
            banner = f'<nav aria-label="Notebook preview"><a href="../examples/">← Example catalog</a> · Executed with Antecedent {release} · Synthetic data</nav>'
            body_start = html.find(">", html.find("<body")) + 1
            html = html[:body_start] + banner + html[body_start:]
            args.output.joinpath(path.stem + ".html").write_text(html, encoding="utf-8")
            measurements.append(
                {
                    "notebook": path.name,
                    "seconds": round(elapsed, 2),
                    "outputs": outputs,
                    "antecedent_version": release,
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            print(f"Executed {path.name}: {elapsed:.1f}s, {outputs} outputs", flush=True)
    args.output.joinpath("execution.json").write_text(
        json.dumps(measurements, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
