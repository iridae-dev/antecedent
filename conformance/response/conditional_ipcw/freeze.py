"""Freeze the executing R survival oracle outputs; run after generate.R."""
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parent
with (root / 'oracle.csv').open() as stream:
    rows = list(csv.DictReader(stream))
columns = {key: [float(row[key]) for row in rows] for key in rows[0]}
expected = {
    'fixture_id': 'response.conditional_ipcw',
    'oracle': 'R survival 3.8.6 coxph Breslow and basehaz left limits',
    'coefficients': [float(x) for x in (root / 'coefficients.txt').read_text().split()],
    'data': columns,
    'atol': 1e-8,
}
# Rust consumes the original singular field names from the CSV.
(root / 'expected.json').write_text(json.dumps(expected, indent=2) + '\n')
