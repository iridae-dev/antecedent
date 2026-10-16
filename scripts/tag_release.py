#!/usr/bin/env python3
import re
from pathlib import Path

text = Path("Cargo.toml").read_text(encoding="utf-8")
m = re.search(r"(?ms)^\[workspace\.package\]\n(.*?)(?=\n\[|\Z)", text)
if not m:
    raise SystemExit("Cargo.toml: [workspace.package] not found")
vm = re.search(r'(?m)^version\s*=\s*"([^"]+)"', m.group(0))
if not vm:
    raise SystemExit("Cargo.toml: version missing under [workspace.package]")
print(vm.group(1))
