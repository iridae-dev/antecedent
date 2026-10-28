"""Freeze the executing R survival oracle outputs; run after generate.R."""
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parent
with (root / 'oracle.csv').open(encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
columns = {key: [float(row[key]) for row in rows] for key in rows[0]}
expected = {
    'fixture_id': 'response.conditional_ipcw',
    # The oracle record (kind, generator, lock, review) is curated in expected.json and is kept.
    'oracle': json.loads((root / 'expected.json').read_text(encoding="utf-8"))['oracle'],
    'command': 'Rscript conformance/response/conditional_ipcw/generate.R && python3 conformance/response/conditional_ipcw/freeze.py',
    'coefficients': [float(x) for x in (root / 'coefficients.txt').read_text(encoding="utf-8").split()],
    'data': columns,
    'atol': 1e-8,
}
# Rust consumes the original singular field names from the CSV.
(root / 'expected.json').write_text(json.dumps(expected, indent=2) + '\n', encoding="utf-8", newline="\n")
