#!/usr/bin/env python3
"""Checker benchmark on a temporary copy of the repository input corpus (not a gate).

Creates synthetic bindings only, never passes or product ledgers. Every binding
uses the whole copied corpus to stress shared input hashing and classification.
The timed checker starts with empty in-memory caches; OS caches are not evicted.
"""
import json
import platform
import shutil
import tempfile
from pathlib import Path

from evidence_earning import binding
from evidence_fingerprint import SKIP_DIRS
from evidence_manifests import ANALYZER, STORE, Checker, canonical, inventory, manifest_id


def main():
    root = Path(__file__).resolve().parents[1]
    count = len(inventory(root))
    with tempfile.TemporaryDirectory(prefix='antecedent-checker-benchmark-') as tmp:
        target = Path(tmp)
        (target/'tools/evidence-analyzer').mkdir(parents=True)
        shutil.copyfile(root/'tools/evidence-analyzer/pin.toml',target/'tools/evidence-analyzer/pin.toml')
        trees = []
        for parent in (root/'crates',):
            for crate in parent.iterdir():
                for sub in ('src','tests'):
                    if (crate/sub).is_dir():
                        trees.append((crate/sub).relative_to(root).as_posix())
        trees += ['python/antecedent','conformance','docs','scripts']
        for rel in trees:
            shutil.copytree(root/rel,target/rel,ignore=shutil.ignore_patterns(*SKIP_DIRS,'*.pyc'))
        prepare = Checker(target)
        inputs = [dict(kind='tree',path=rel,digest=prepare.input_digest('tree',rel),role='external') for rel in trees]
        records = []
        for index in range(count):
            record = binding('scientific',f'engineering-benchmark-{index}','synthetic')
            manifest = dict(schema=1,normalization='raw-bytes-v1',record=record,
                compiler={k:prepare.pin[k] for k in ('release','commit')},
                analyzer=ANALYZER,inputs=inputs,fallbacks=[])
            ref=manifest_id(manifest)
            path=target/STORE/f'{ref}.json'
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(canonical(manifest))
            records.append((record,ref))
        checker=Checker(target)
        report=checker.report(records)
        elapsed=report['timings']['total_seconds']
        print(json.dumps(dict(host=platform.platform(),python=platform.python_version(),
            synthetic=True,records=count,current=report['counts']['current'],
            corpus_trees=len(trees),timings=report['timings'],
            target_seconds=5,within_target=elapsed<=5),indent=2))
        return int(elapsed>5 or report['counts']['current']!=count)

if __name__=='__main__':
    raise SystemExit(main())
