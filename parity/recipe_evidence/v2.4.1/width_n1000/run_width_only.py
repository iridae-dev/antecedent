"""Run the two predeclared width supplements once; never evaluate coverage."""
from pathlib import Path
import csv
import datetime
import hashlib
import json
import math
import subprocess
import time

HERE = Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def upper_median(values):
    return sorted(values)[len(values) // 2]

plan_path = HERE / 'plan.json'
plan = json.loads(plan_path.read_text(encoding='utf-8'))
assert plan['status'] == 'SEALED_PLAN_NO_SIMULATIONS'
assert digest(plan_path) == (HERE / 'plan.sha256').read_text(encoding='utf-8').split()[0]
assert digest(HERE / 'width_only.rs') == plan['helper_sha256']
assert digest(HERE / 'observational_effects.rs') == plan['dgp']['sha256']
assert digest(Path(__file__)) == plan['orchestration_sha256']
for path, expected in plan['artifact_sha256'].items():
    assert digest(Path(path)) == expected, path
assert not (HERE / 'width_only').exists(), 'compile/run receipt already exists; no overwrite'
with (HERE / 'compile.log').open('x', encoding='utf-8') as log:
    subprocess.run(plan['compiler_command'], stdout=log, stderr=subprocess.STDOUT, check=True)
binary = HERE / 'width_only'
compiled = {'plan_sha256': digest(plan_path), 'binary_sha256': digest(binary),
            'compiled_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
(HERE / 'compiled-artifact.json').write_text(json.dumps(compiled, indent=2)+'\n', encoding='utf-8')
results = []
for case, base in [('policy', 0x6b0000), ('survival', 0x5e0000)]:
    raw = HERE / (case + '.csv')
    summary = HERE / (case + '.summary.json')
    started = time.monotonic()
    with raw.open('x', encoding='utf-8') as stdout, summary.open('x', encoding='utf-8') as stderr:
        status = subprocess.run([str(binary), case], stdout=stdout, stderr=stderr).returncode
    with raw.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2048
    valid = []
    for i, row in enumerate(rows):
        assert row['case'] == case and int(row['replicate']) == i
        assert int(row['seed']) == ((base+i) ^ int(plan['seed_salt_hex'], 16))
        assert row['status'] in {'reported', 'refused'}
        if row['status'] == 'reported':
            width, truth = float(row['width']), float(row['abs_truth'])
            assert math.isfinite(width) and width >= 0 and math.isfinite(truth) and truth > 0
            valid.append((width, truth))
        else:
            assert row['width'] == '' and row['abs_truth'] == ''
    widths, truths = zip(*valid)
    independent = {'reported':len(valid), 'attempts':2048,
                   'median_width':upper_median(widths), 'median_abs_truth':upper_median(truths),
                   'median_per_replicate_width_over_abs_truth':upper_median([w/t for w,t in valid])}
    independent['median_width_over_median_abs_truth'] = independent['median_width']/independent['median_abs_truth']
    independent['reporting_rate'] = len(valid)/2048
    results.append({'case':case, 'exit_code':status, 'elapsed_seconds':time.monotonic()-started,
                    'raw_sha256':digest(raw), 'summary_sha256':digest(summary),
                    'independently_recomputed_width_metrics':independent,
                    'passes_frozen_width_and_reporting':status==0 and len(valid)>=2000 and
                     independent['median_width_over_median_abs_truth']<=1.5 and
                     independent['median_per_replicate_width_over_abs_truth']<=1.5})
    print(json.dumps(results[-1]), flush=True)
receipt = {'plan_sha256':digest(plan_path), 'compiled_artifact':compiled, 'results':results,
           'coverage_computed':False, 'coverage_records_changed':False}
with (HERE / 'results.json').open('x', encoding='utf-8') as handle:
    json.dump(receipt, handle, indent=2)
    handle.write('\n')
assert all(r['passes_frozen_width_and_reporting'] for r in results)
