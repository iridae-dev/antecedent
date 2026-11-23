"""Enumerate target-scoped Cargo dependencies for a manual Nix build audit.

This offline inventory is a conservative review aid, not an access monitor or
evidence that a package is safe. Cargo tree includes the package's dev graph and
can include packages not compiled by a particular test target. In particular,
the inventory neither checks actual compiler units nor verifies the Nix builder,
Cargo registry contents, caller-generated macro input, or transitive OS reads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tomllib
from pathlib import Path

from _repo import REPO_ROOT, scientific_target_feature_args
from evidence_earning import require_compile_environment

NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")
TRIPLE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")
TREE_LINE = re.compile(r"^([A-Za-z0-9_-]+) v([^\s]+)(?:\s|$)")
# Search hints only. Absence is not proof: imports, generated tokens, macros,
# subprocesses and native code may all perform reads without matching these.
PATTERNS = {
    'temp': re.compile(r'/private/tmp|/tmp/|TMPDIR|temp_dir|tempfile', re.I),
    'env': re.compile(r'(?:env::(?:var|var_os|vars)|(?:option_)?env!\s*\(|getenv\s*\()'),
    'file': re.compile(r'(?:std::fs|\bfs::|File::open|read_to_string|read_to_end|include_(?:str|bytes)!\s*\()'),
    'process': re.compile(r'(?:std::process|Command::new|exec\(|posix_spawn|system\s*\()'),
    'nondeterminism': re.compile(r'(?:getrandom|thread_rng|SystemTime|clock_gettime|/dev/(?:u?random))'),
}


def tree_packages(text: str) -> tuple[dict[tuple[str, str], set[str]], dict[tuple[str, str], set[tuple[str, str]]]]:
    """Package versions, feature union and parent edges from Cargo's depth tree."""
    found: dict[tuple[str, str], set[str]] = {}
    parents: dict[tuple[str, str], set[tuple[str, str]]] = {}
    stack: list[tuple[str, str]] = []
    for line in text.splitlines():
        depth_text, marker, row = line.partition('|')
        package, sep, features = row.partition('\t')
        match = TREE_LINE.match(package)
        if not marker or not depth_text.isdecimal() or not sep or not match:
            raise ValueError(f'unrecognized cargo tree line: {line!r}')
        depth = int(depth_text)
        if depth > len(stack) or (depth == 0 and found):
            raise ValueError('invalid Cargo tree depth')
        del stack[depth:]
        key = match.groups()
        if depth:
            parents.setdefault(key, set()).add(stack[-1])
        stack.append(key)
        # Cargo appends its subtree-deduplication marker after the final
        # formatted field, not necessarily after the package name.
        features = features.removesuffix(' (*)')
        found.setdefault(key, set()).update(f for f in features.split(',') if f)
    if not found:
        raise ValueError('empty Cargo target graph')
    return found, parents


def source_hints(package_dir: Path) -> dict[str, list[str]]:
    """Candidate source locations; matches include false positives and omissions."""
    result = {key: [] for key in PATTERNS}
    for path in sorted(package_dir.rglob('*.rs')):
        if any(part in ('target', '.git') for part in path.relative_to(package_dir).parts):
            continue
        if path.is_symlink():
            raise ValueError(f'linked package source: {path}')
        try:
            lines = path.read_text(encoding='utf-8').splitlines()
        except (OSError, UnicodeError) as error:
            raise ValueError(f'cannot inspect source: {path}') from error
        for number, line in enumerate(lines, 1):
            for key, pattern in PATTERNS.items():
                if pattern.search(line):
                    result[key].append(f'{path.relative_to(package_dir).as_posix()}:{number}')
    return {key: value for key, value in result.items() if value}


def compiler_units(metadata: dict, graph: dict) -> dict:
    """Normalize Cargo's unstable unit graph without importing host paths.

    This is exact for the Cargo command and environment that generated it, not
    an attestation that a later Nix build used that graph.
    """
    if graph.get('version') != 1 or not isinstance(graph.get('units'), list) or not isinstance(graph.get('roots'), list):
        raise ValueError('unsupported Cargo unit graph')
    units = graph['units']
    by_identity: dict[tuple[str, str, str | None], dict] = {}
    by_cargo_id: dict[str, dict] = {}
    for package in metadata.get('packages', []):
        if isinstance(package.get('id'), str):
            if package['id'] in by_cargo_id:
                raise ValueError('ambiguous Cargo metadata package ID')
            by_cargo_id[package['id']] = package
        origin = package.get('source') or 'path+' + Path(package['manifest_path']).parent.resolve().as_uri()
        key = (package['name'], package['version'], origin)
        if key in by_identity:
            raise ValueError('ambiguous Cargo metadata package identity')
        by_identity[key] = package
    output = []
    for unit in units:
        pkg_id = unit.get('pkg_id', '')
        package = by_cargo_id.get(pkg_id)
        if package is None:
            # Cargo 1.85's graph uses a legacy human-readable package ID;
            # newer versions emit the same canonical ID as `cargo metadata`.
            match = re.fullmatch(r'([^ ]+) ([^ ]+) \((.+)\)', pkg_id)
            if not match or match.groups() not in by_identity:
                raise ValueError('unresolvable Cargo compiler-unit package')
            package = by_identity[match.groups()]
        name, version = package['name'], package['version']
        target = unit.get('target')
        deps = unit.get('dependencies')
        features = unit.get('features')
        if not isinstance(target, dict) or not isinstance(deps, list) or not isinstance(features, list):
            raise ValueError('invalid Cargo compiler unit')
        unit_dependencies = []
        for dep in deps:
            index = dep.get('index') if isinstance(dep, dict) else None
            if type(index) is not int or not 0 <= index < len(units):
                raise ValueError('invalid Cargo compiler-unit dependency')
            # Alias, public/noprelude and future Cargo fields can affect how
            # a dependency is used; do not collapse edges to indices alone.
            unit_dependencies.append(dict(dep))
        if not all(isinstance(f, str) for f in features):
            raise ValueError('invalid Cargo compiler-unit features')
        try:
            target_path = Path(target['src_path']).resolve().relative_to(Path(package['manifest_path']).parent.resolve()).as_posix()
        except (KeyError, ValueError) as error:
            raise ValueError('compiler-unit target is outside its package') from error
        output.append({'name': name, 'version': version, 'mode': unit.get('mode'),
                       'platform': unit.get('platform'), 'target_name': target.get('name'),
                       'target_kind': target.get('kind'), 'target_path': target_path,
                       'target_details': {k: v for k, v in target.items() if k not in ('name', 'kind', 'src_path')},
                       'profile': unit.get('profile'),
                       'unit_details': {k: v for k, v in unit.items()
                                        if k not in ('pkg_id', 'mode', 'platform', 'target', 'profile', 'features', 'dependencies')},
                       'features': sorted(features),
                       'dependencies': sorted(unit_dependencies, key=lambda d: json.dumps(d, sort_keys=True)),
                       'audit_status': 'unreviewed'})
    roots = graph['roots']
    if not roots or any(type(i) is not int or not 0 <= i < len(units) for i in roots):
        raise ValueError('invalid Cargo compiler-unit roots')
    return {'roots': roots, 'units': output}


def compare_compiler_units(host: dict, builder: dict) -> dict:
    """Compare normalized unit identities and dependency edges, not index order."""
    def keys(graph: dict) -> set[str]:
        units = graph['units']
        def identity(unit: dict) -> str:
            return json.dumps({k: v for k, v in unit.items() if k not in ('dependencies', 'audit_status')},
                              sort_keys=True, separators=(',', ':'))
        identifiers = [identity(unit) for unit in units]
        rows = set()
        for index, unit in enumerate(units):
            dependencies = [{**dep, 'index': identifiers[dep['index']]} for dep in unit['dependencies']]
            rows.add(json.dumps({'unit': identifiers[index],
                                 'dependencies': sorted(dependencies, key=lambda d: json.dumps(d, sort_keys=True))},
                                sort_keys=True, separators=(',', ':')))
        return rows
    local, built = keys(host), keys(builder)
    return {'match': local == built, 'host_only_unit_edges': len(local - built),
            'builder_only_unit_edges': len(built - local),
            'interpretation': 'comparison of caller-supplied unit graphs; not builder or sandbox attestation'}


def inventory(root: Path, metadata: dict, tree: str, crate: str, target: str,
              features: list[str], *, unit_graph: dict | None = None,
              test_name: str | None = None, hints: bool = False) -> dict:
    packages, parents = tree_packages(tree)
    by_name: dict[tuple[str, str], list[dict]] = {}
    for package in metadata.get('packages', []):
        by_name.setdefault((package['name'], package['version']), []).append(package)
    lock_bytes = (root / 'Cargo.lock').read_bytes()
    lock = tomllib.loads(lock_bytes.decode('utf-8'))
    locked = {(p['name'], p['version'], p.get('source')): p.get('checksum') for p in lock['package']}
    rows = []
    for key, active_features in sorted(packages.items()):
        matches = by_name.get(key, [])
        if len(matches) != 1:
            raise ValueError(f'Cargo package cannot be resolved uniquely: {key}')
        p = matches[0]
        kinds = sorted({kind for target_ in p['targets'] for kind in target_['kind']
                        if kind in ('proc-macro', 'custom-build')})
        source = p.get('source')
        checksum = locked.get((p['name'], p['version'], source))
        if source and (not source.startswith('registry+') or not isinstance(checksum, str)
                       or not re.fullmatch(r'[0-9a-f]{64}', checksum)):
            raise ValueError(f'unsupported or unlocked package source: {p["name"]} {p["version"]}')
        row = {'name': p['name'], 'version': p['version'], 'source': source or 'workspace',
               'checksum': checksum, 'compile_time_kinds': kinds,
               'cargo_tree_features': sorted(active_features),
               'parents': [{'name': name, 'version': version} for name, version in sorted(parents.get(key, set()))],
               'audit_status': 'unreviewed'}
        if source is None:
            manifest = Path(p['manifest_path']).resolve()
            try:
                manifest_rel = manifest.relative_to(root.resolve()).as_posix()
            except ValueError as error:
                raise ValueError(f'workspace package is outside repository: {p["name"]}') from error
            row['manifest'] = manifest_rel
            row['manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if hints:
            row['source_search_hints'] = source_hints(Path(p['manifest_path']).parent)
        rows.append(row)
    result = {'schema': 1, 'crate': crate, 'target_triple': target, 'test_target': test_name,
            'requested_features': sorted(features),
            'cargo_lock_sha256': hashlib.sha256(lock_bytes).hexdigest(),
            'interpretation': 'Cargo unit graph is exact only for this host command; Nix build graph and source safety remain unverified',
            'packages': rows}
    if unit_graph is not None:
        result['compiler_units'] = compiler_units(metadata, unit_graph)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--crate', required=True)
    parser.add_argument('--target-triple', required=True)
    parser.add_argument('--test', help='Cargo integration target; omit for the library')
    parser.add_argument('--hints', action='store_true', help='scan all package Rust sources for review candidates')
    parser.add_argument('--builder-metadata', type=Path, help='raw Cargo metadata JSON from the Nix builder')
    parser.add_argument('--builder-unit-graph', type=Path, help='raw Cargo unit-graph JSON from the Nix builder')
    args = parser.parse_args()
    if not NAME.fullmatch(args.crate) or not TRIPLE.fullmatch(args.target_triple) or (args.test and not NAME.fullmatch(args.test)):
        parser.error('invalid Cargo crate, target triple, or test name')
    if bool(args.builder_metadata) != bool(args.builder_unit_graph):
        parser.error('both builder JSON files must be supplied together')
    require_compile_environment(REPO_ROOT)
    manifest = REPO_ROOT / 'crates' / args.crate / 'Cargo.toml'
    target_args = ['--test', args.test] if args.test else ['--lib']
    feature_args = scientific_target_feature_args(manifest, target_args)
    features = feature_args[1].split(',') if feature_args else []
    version = subprocess.check_output(['rustc', '-vV'], cwd=REPO_ROOT, text=True)
    rustc_info = dict(line.split(': ', 1) for line in version.splitlines() if ': ' in line)
    pin = tomllib.loads((REPO_ROOT / 'tools/evidence-analyzer/pin.toml').read_text(encoding="utf-8"))
    cargo_version = subprocess.check_output(['cargo', '-V'], cwd=REPO_ROOT, text=True).strip()
    if (rustc_info.get('host') != args.target_triple or rustc_info.get('release') != pin['release']
            or rustc_info.get('commit-hash') != pin['commit']
            or not cargo_version.startswith('cargo ' + pin['release'] + ' ')):
        raise ValueError('unit inventory requires the pinned compiler and native host target')
    metadata = json.loads(subprocess.check_output(['cargo', 'metadata', '--offline', '--locked', '--format-version', '1'], cwd=REPO_ROOT))
    cmd = ['cargo', 'tree', '--offline', '--locked', '-p', args.crate, '--target', args.target_triple,
           '-e', 'normal,build,dev', '--prefix', 'depth', '--format', '|{p}\t{f}', *feature_args]
    tree = subprocess.check_output(cmd, cwd=REPO_ROOT, text=True)
    # The unstable graph option only prints metadata; it does not build or run.
    # Cargo 1.85.1 needs this opt-in. It is confined to this diagnostic process.
    env = {**os.environ, 'RUSTC_BOOTSTRAP': '1', 'CARGO_PROFILE_TEST_OPT_LEVEL': '2',
           'CARGO_PROFILE_TEST_DEBUG': 'line-tables-only', 'RUSTFLAGS': '-C instrument-coverage'}
    unit_cmd = ['cargo', '-Z', 'unstable-options', 'test', '--unit-graph', '--no-run', '--offline', '--locked',
                '-p', args.crate, *target_args, *feature_args]
    graph = json.loads(subprocess.check_output(unit_cmd, cwd=REPO_ROOT, env=env))
    result = inventory(REPO_ROOT, metadata, tree, args.crate, args.target_triple, features,
                       unit_graph=graph, test_name=args.test, hints=args.hints)
    result['rustc'] = {'release': rustc_info['release'], 'commit': rustc_info['commit-hash'], 'host': rustc_info['host']}
    result['cargo_version'] = cargo_version
    result['graph_environment'] = {'RUSTC_BOOTSTRAP': '1 (Cargo graph query only)',
                                   'CARGO_PROFILE_TEST_OPT_LEVEL': '2',
                                   'CARGO_PROFILE_TEST_DEBUG': 'line-tables-only',
                                   'RUSTFLAGS': '-C instrument-coverage'}
    result['workspace_inputs_sha256'] = {
        name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest()
        if (REPO_ROOT / name).is_file() else None
        for name in ('Cargo.toml', 'rust-toolchain.toml', '.cargo/config', '.cargo/config.toml')
    }
    if args.builder_metadata:
        builder_metadata = json.loads(args.builder_metadata.read_text(encoding="utf-8"))
        builder_graph = json.loads(args.builder_unit_graph.read_text(encoding="utf-8"))
        result['builder_compiler_units'] = compiler_units(builder_metadata, builder_graph)
        result['unit_graph_comparison'] = compare_compiler_units(result['compiler_units'], result['builder_compiler_units'])
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
