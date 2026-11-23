#!/usr/bin/env python3
import os
import sys

import antecedent

module_file = getattr(antecedent, "__file__", None)
expected_root = os.path.realpath(sys.argv[1])
try:
    inside = bool(module_file) and os.path.commonpath(
        (os.path.realpath(module_file), expected_root)
    ) == expected_root
except (TypeError, ValueError):
    inside = False
if not inside:
    raise SystemExit(f"FAIL: antecedent imported from {module_file}, not the wheel")
print("installed wheel:", antecedent.__version__, module_file)
