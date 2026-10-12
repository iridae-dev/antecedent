#!/usr/bin/env python3
import os
import sys

import antecedent

if not os.path.realpath(antecedent.__file__).startswith(os.path.realpath(sys.argv[1])):
    raise SystemExit(f"FAIL: antecedent imported from {antecedent.__file__}, not the wheel")
print("installed wheel:", antecedent.__version__, antecedent.__file__)
