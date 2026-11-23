#!/usr/bin/env python3
"""Build and validate the driver-free CPU wheel and the separate accelerator distribution.

The CPU wheel is the standard wheel: it links no device library. The accelerator
distribution is the same package built with the ``accelerator-wgpu`` feature and relabelled
with the PEP 440 local version ``+wgpu``, so it replaces the CPU wheel in an environment and
can never be mistaken for it.

Subcommands:

``build``      maturin-build both wheels into ``--out`` (CPU, then accelerator relabelled).
``inspect``    static checks of one wheel: version label, linked system libraries, device
               library symbols.
``venv``       install one wheel into a fresh virtual environment and run the probe there.
``all``        build, inspect, install each wheel in its own clean environment, run the probe
               (including the hidden-device refusal and cross-consumption of each wheel's
               exported model by the other), and write ``--report``.
``selftest``   exercise the relabeller and the inspector on a synthetic wheel (no build).

Nothing here touches the repository's own ``antecedent`` import path: the probe runs from the
work directory with ``python -I`` inside the new environment.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE = REPO_ROOT / "scripts" / "_distribution_probe.py"
ACCELERATOR_LOCAL = "wgpu"
ACCELERATOR_FEATURE = "accelerator-wgpu"

# A driver-free wheel must not load any of these. Matching is by substring of the library name.
DEVICE_LIBRARIES = (
    "Metal",
    "QuartzCore",
    "IOSurface",
    "libvulkan",
    "vulkan-1",
    "libGL",
    "libEGL",
    "libcuda",
    "libnvidia",
    "libOpenCL",
    "d3d12",
)
# Symbols of the device library. Present in the accelerator build, absent from the CPU build.
DEVICE_SYMBOLS = (b"wgpu_core", b"wgpu_hal")


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    """Run ``cmd`` and return stdout. A failing command raises with its output."""
    done = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True, check=False)  # noqa: S603
    if done.returncode != 0:
        raise SystemExit(
            f"command failed ({done.returncode}): {shlex.join(cmd)}\n{done.stdout}\n{done.stderr}"
        )
    return done.stdout


def wheel_version(wheel: Path) -> str:
    with zipfile.ZipFile(wheel) as zf:
        meta = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
        for line in zf.read(meta).decode("utf-8").splitlines():
            if line.startswith("Version: "):
                return line.removeprefix("Version: ").strip()
    raise SystemExit(f"{wheel.name}: no Version in METADATA")


def record_line(path: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{path},sha256={digest},{len(data)}"


def relabel_wheel(src: Path, out_dir: Path, local: str = ACCELERATOR_LOCAL) -> Path:
    """Copy ``src`` with version ``V`` relabelled ``V+local``; RECORD is regenerated."""
    version = wheel_version(src)
    if "+" in version:
        raise SystemExit(f"{src.name} already carries a local version")
    new_version = f"{version}+{local}"
    name = src.name.split("-", 1)[0]
    old_info, new_info = (
        f"{name}-{version}.dist-info",
        f"{name}-{new_version}.dist-info",
    )
    out = out_dir / src.name.replace(f"-{version}-", f"-{new_version}-", 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[str] = []
    with (
        zipfile.ZipFile(src) as zin,
        zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout,
    ):
        for item in zin.infolist():
            if item.filename == f"{old_info}/RECORD":
                continue
            data = zin.read(item.filename)
            target = item.filename
            if target.startswith(f"{old_info}/"):
                target = f"{new_info}/{target.removeprefix(old_info + '/')}"
            if target == f"{new_info}/METADATA":
                text = data.decode("utf-8").replace(
                    f"Version: {version}\n", f"Version: {new_version}\n", 1
                )
                data = text.encode("utf-8")
            clone = zipfile.ZipInfo(target, date_time=item.date_time)
            clone.compress_type = zipfile.ZIP_DEFLATED
            clone.external_attr = item.external_attr
            zout.writestr(clone, data)
            records.append(record_line(target, data))
        records.append(f"{new_info}/RECORD,,")
        zout.writestr(f"{new_info}/RECORD", "\n".join(records) + "\n")
    return out


def native_module(zf: zipfile.ZipFile) -> str:
    return next(n for n in zf.namelist() if "/_native" in n and n.endswith((".so", ".pyd")))


def linked_libraries(blob: bytes, suffix: str) -> list[str] | None:
    """System libraries the native module loads, or ``None`` when no tool can tell."""
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / f"native{suffix}"
        path.write_bytes(blob)
        system = platform.system()
        try:
            if system == "Darwin" and shutil.which("otool"):
                lines = run(["otool", "-L", str(path)]).splitlines()[1:]
                return [line.split()[0] for line in lines if line.strip()]
            if system == "Linux" and shutil.which("readelf"):
                out = run(["readelf", "-d", str(path)])
                return [
                    line.split("[", 1)[1].rstrip("]")
                    for line in out.splitlines()
                    if "(NEEDED)" in line and "[" in line
                ]
        except SystemExit:
            return None
    return None


def inspect_wheel(wheel: Path, expect: str) -> list[str]:
    """Problems with ``wheel`` as the ``cpu`` or ``accelerator`` distribution."""
    problems: list[str] = []
    version = wheel_version(wheel)
    with zipfile.ZipFile(wheel) as zf:
        member = native_module(zf)
        blob = zf.read(member)
    has_device = any(symbol in blob for symbol in DEVICE_SYMBOLS)
    if expect == "cpu":
        if "+" in version:
            problems.append(f"cpu wheel carries a local version: {version}")
        if has_device:
            problems.append("cpu wheel contains the device library")
        libs = linked_libraries(blob, Path(member).suffix)
        if libs is not None:
            for lib in libs:
                if any(token in lib for token in DEVICE_LIBRARIES):
                    problems.append(f"cpu wheel links a device library: {lib}")
    else:
        if not version.endswith(f"+{ACCELERATOR_LOCAL}"):
            problems.append(f"accelerator wheel version is not labelled +{ACCELERATOR_LOCAL}")
        if not has_device:
            problems.append("accelerator wheel does not contain the device library")
        libs = linked_libraries(blob, Path(member).suffix)
        if (
            libs is not None
            and platform.system() == "Darwin"
            and not any("Metal" in lib for lib in libs)
        ):
            problems.append("accelerator wheel does not link Metal on macOS")
    return problems


def maturin_command(maturin: str) -> list[str]:
    cmd = shlex.split(maturin)
    if shutil.which(cmd[0]) is None:
        raise SystemExit(f"{cmd[0]} not found; pass --maturin 'uvx maturin' or install maturin")
    return cmd


def build_wheel(maturin: str, out: Path, *, accelerator: bool) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.whl"):
        stale.unlink()
    before = set(out.glob("*.whl"))
    cmd = [
        *maturin_command(maturin),
        "build",
        "--release",
        "--manifest-path",
        str(REPO_ROOT / "python" / "Cargo.toml"),
        "--out",
        str(out),
    ]
    if accelerator:
        cmd += ["--features", ACCELERATOR_FEATURE]
    run(cmd, cwd=REPO_ROOT)
    built = sorted(set(out.glob("*.whl")) - before)
    if len(built) != 1:
        raise SystemExit(f"expected one new wheel in {out}, found {[p.name for p in built]}")
    return built[0]


def build_all(maturin: str, out: Path) -> tuple[Path, Path]:
    """Return ``(cpu_wheel, accelerator_wheel)``. The accelerator wheel is relabelled."""
    cpu = build_wheel(maturin, out / "cpu", accelerator=False)
    raw = build_wheel(maturin, out / "accelerator-raw", accelerator=True)
    return cpu, relabel_wheel(raw, out / "accelerator")


def venv_probe(
    wheel: Path,
    expect: str,
    work: Path,
    *,
    env_extra: dict[str, str] | None = None,
    artifact_out: Path | None = None,
    artifact_in: Path | None = None,
    index_args: list[str] | None = None,
) -> dict[str, object]:
    """Install ``wheel`` in a fresh environment and run the probe there."""
    work = work.resolve()
    venv = work / f"venv-{expect}-{'hidden' if env_extra else 'visible'}-{os.getpid()}"
    if venv.exists():
        shutil.rmtree(venv)
    run([sys.executable, "-m", "venv", str(venv)])
    py = venv / ("Scripts" if os.name == "nt" else "bin") / "python"
    run(
        [str(py), "-m", "pip", "install", "--no-input", "--disable-pip-version-check",
         *(index_args or []), str(wheel)]
    )  # fmt: skip
    cmd = [
        str(py),
        "-I",
        str(PROBE),
        "--expect",
        expect,
        "--version",
        wheel_version(wheel),
    ]
    if artifact_out:
        cmd += ["--artifact-out", str(artifact_out)]
    if artifact_in:
        cmd += ["--artifact-in", str(artifact_in)]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.pop("ANTECEDENT_DISABLE_DEVICE", None)
    env.update(env_extra or {})
    done = subprocess.run(cmd, cwd=work, env=env, text=True, capture_output=True, check=False)  # noqa: S603
    try:
        report = json.loads(done.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        report = {"ok": False, "checks": [], "error": done.stdout + done.stderr}
    report["returncode"] = done.returncode
    report["wheel"] = wheel.name
    report["hidden_device"] = bool(env_extra)
    if done.returncode != 0:
        report["stderr"] = done.stderr[-2000:]
    return report


def acceptance(
    cpu: Path, accelerator: Path, work: Path, index_args: list[str]
) -> dict[str, object]:
    problems = {
        "cpu": inspect_wheel(cpu, "cpu"),
        "accelerator": inspect_wheel(accelerator, "accelerator"),
    }
    work = work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    cpu_artifact, acc_artifact = (
        work / "cpu_model.json",
        work / "accelerator_model.json",
    )
    runs = [
        venv_probe(cpu, "cpu", work, artifact_out=cpu_artifact, index_args=index_args),
        venv_probe(accelerator, "accelerator", work, artifact_out=acc_artifact, index_args=index_args),
        venv_probe(
            accelerator, "accelerator", work, env_extra={"ANTECEDENT_DISABLE_DEVICE": "1"},
            artifact_in=cpu_artifact, index_args=index_args,
        ),
        venv_probe(cpu, "cpu", work, artifact_in=acc_artifact, index_args=index_args),
    ]  # fmt: skip
    ok = not any(problems.values()) and all(r.get("ok") for r in runs)
    return {"ok": ok, "inspection": problems, "runs": runs, "host": platform.platform()}


def selftest() -> int:
    """Relabel a synthetic wheel and check that the result is a coherent, labelled wheel."""
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        src = root / "antecedent-1.2.3-cp313-cp313-macosx_11_0_arm64.whl"
        info = "antecedent-1.2.3.dist-info"
        with zipfile.ZipFile(src, "w") as zf:
            zf.writestr("antecedent/__init__.py", "x = 1\n")
            native = zipfile.ZipInfo("antecedent/_native.cpython-313-darwin.so")
            native.external_attr = 0o755 << 16
            zf.writestr(native, b"\x00native wgpu_core\x00")
            zf.writestr(
                f"{info}/METADATA",
                "Metadata-Version: 2.1\nName: antecedent\nVersion: 1.2.3\n",
            )
            zf.writestr(f"{info}/WHEEL", "Wheel-Version: 1.0\n")
            zf.writestr(f"{info}/RECORD", "stale\n")
        assert wheel_version(src) == "1.2.3"
        out = relabel_wheel(src, root / "out")
        assert out.name == "antecedent-1.2.3+wgpu-cp313-cp313-macosx_11_0_arm64.whl", out.name
        assert wheel_version(out) == "1.2.3+wgpu"
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
            assert not any(n.startswith("antecedent-1.2.3.dist-info/") for n in names)
            record = zf.read("antecedent-1.2.3+wgpu.dist-info/RECORD").decode()
            for name in names:
                if name.endswith("/RECORD"):
                    continue
                assert record_line(name, zf.read(name)) in record, name
            assert (
                zf.getinfo("antecedent/_native.cpython-313-darwin.so").external_attr >> 16 == 0o755
            )
            assert zf.testzip() is None
        assert inspect_wheel(out, "accelerator") == [] or platform.system() == "Darwin"
        assert any("device library" in p for p in inspect_wheel(src, "cpu"))
        try:
            relabel_wheel(out, root / "again")
        except SystemExit:
            pass
        else:
            raise AssertionError("a relabelled wheel must not be relabelled again")
    print("distribution acceptance self-test: ok")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "all"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--maturin", default="uvx maturin")
    all_p = sub.choices["all"]
    all_p.add_argument("--report", type=Path, required=True)
    all_p.add_argument("--cpu-wheel", type=Path)
    all_p.add_argument("--accelerator-wheel", type=Path)
    all_p.add_argument("--pip-arg", action="append", default=[], help="extra pip install argument")
    ins = sub.add_parser("inspect")
    ins.add_argument("wheel", type=Path)
    ins.add_argument("--expect", choices=("cpu", "accelerator"), required=True)
    venv = sub.add_parser("venv")
    venv.add_argument("wheel", type=Path)
    venv.add_argument("--expect", choices=("cpu", "accelerator"), required=True)
    venv.add_argument("--work", type=Path, required=True)
    venv.add_argument("--pip-arg", action="append", default=[])
    sub.add_parser("selftest")
    args = parser.parse_args()

    if args.command == "selftest":
        return selftest()
    if args.command == "build":
        cpu, accelerator = build_all(args.maturin, args.out)
        print(cpu, accelerator, sep="\n")
        return 0
    if args.command == "inspect":
        problems = inspect_wheel(args.wheel, args.expect)
        print("\n".join(problems) or f"{args.wheel.name}: ok as {args.expect}")
        return 1 if problems else 0
    if args.command == "venv":
        report = venv_probe(args.wheel, args.expect, args.work, index_args=args.pip_arg)
        print(json.dumps(report, indent=2))
        return 0 if report.get("ok") else 1
    out = args.out
    if args.cpu_wheel and args.accelerator_wheel:
        cpu, accelerator = args.cpu_wheel, args.accelerator_wheel
    else:
        cpu, accelerator = build_all(args.maturin, out)
    result = acceptance(cpu, accelerator, out / "work", args.pip_arg)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"ok": result["ok"], "report": str(args.report)}))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
