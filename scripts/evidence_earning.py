"""Snapshot and publish dependency manifests for evidence that has just been earned.

Dependency discovery runs at earning time; `evidence_manifests.Checker` later
checks the stored inputs without Cargo metadata or a compiler. Earning refuses a
non-canonical build environment and untracked files among the inputs, so a
manifest written on a development machine is current in a clean checkout.
A Rust assertion's sources are reduced to the items its execution coverage and
the compiler inventory show it depends on (`select_items`, `evidence_items.py`);
otherwise complete source is kept.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from _repo import load_toml
from compile_time_environment import noncanonical_compile_environment
from evidence_fingerprint import INPUT_DECLARATIONS, Fingerprinter, closure, locate
from evidence_manifests import (
    ANALYZER,
    BUILD_ENV,
    INPUT_GROUPS,
    NORMALIZATION,
    SCHEMA,
    STORE,
    Checker,
    canonical,
    digest,
    grouped,
    manifest_id,
    noncanonical_environment,
)


def binding(kind: str, key: str, fingerprint: str, measurement=None) -> dict:
    return dict(kind=kind, key=key, fingerprint=fingerprint, measurement=measurement)


def probe_binding(row: dict) -> dict:
    result = binding('probe', row.get('record_id'), row.get('fingerprint'), row.get('calibration_sha'))
    result.update({k: row.get(k) for k in ('sha', 'k', 'bound', 'result')})
    if 'measurement_digest' in row:
        result['measurement_digest'] = row['measurement_digest']
    return result


def measurement_digest(record: dict) -> str:
    from collect_coverage_records import FIELDS, OPTIONAL_FIELDS
    persisted = set(FIELDS) | set(OPTIONAL_FIELDS)
    return digest(canonical({k: v for k, v in record.items() if k in persisted and v is not None}))


def measurement_binding(record: dict) -> dict:
    return {**binding('calibration', record.get('id'), record.get('fingerprint'), record.get('calibration_sha')),
            'measurement_digest': measurement_digest(record)}


def declarations(row: dict) -> dict:
    return {k: row.get(k) for k in ('evidence_test', 'evidence_assertion')} | {
        'fixtures': sorted(row.get('fixtures', [])), 'roles': sorted(row.get('roles', []))}


_COMPILE_TIME: dict[Path, dict[str, set[str]] | None] = {}


def require_compile_environment(root: Path) -> None:
    """Reject known ambient overrides before discovering or executing a build."""
    issues = noncanonical_compile_environment(root, os.environ)
    # BUILD_ENV has its own canonical-value policy (including allowed lint-only
    # flags). Report names only, since environment values can contain secrets.
    issues += ['environment override ' + issue.split('=', 1)[0]
               for issue in noncanonical_environment(os.environ, [])]
    if issues:
        raise RuntimeError('unassessed compile-time environment; unset overrides or remove '
                           'ambient Cargo config before earning: ' + ', '.join(issues))


def compile_time_code(root: Path) -> dict[str, set[str]] | None:
    """Workspace package -> the packages with a procedural macro or build script in its
    resolved dependency graph (itself included), from the full Cargo metadata; `None` when
    it cannot be read. Cached per workspace for one process."""
    if root not in _COMPILE_TIME:
        found = None
        for flags in (["--offline", "--locked"], []):
            out = subprocess.run(["cargo", "metadata", "--format-version", "1", *flags], cwd=root,
                                 capture_output=True, text=True)
            if out.returncode != 0:
                continue
            import json
            data = json.loads(out.stdout)
            packages = {p["id"]: p for p in data["packages"]}
            compile_time = {pid for pid, p in packages.items()
                            if any(k in ("proc-macro", "custom-build") for t in p.get("targets", []) for k in t.get("kind", []))}
            edges = {n["id"]: [d["pkg"] for d in n.get("deps", [])] for n in (data.get("resolve") or {}).get("nodes", [])}
            found = {}
            for member in data.get("workspace_members", []):
                seen, todo = set(), [member]
                while todo:
                    pid = todo.pop()
                    if pid not in seen:
                        seen.add(pid)
                        todo.extend(edges.get(pid, []))
                found[packages[member]["name"]] = {packages[pid]["name"] for pid in seen & compile_time}
            break
        _COMPILE_TIME[root] = found
    return _COMPILE_TIME[root]


_HOST_UNITS: dict[tuple, bool] = {}


def _registry_assessed(root: Path, info: dict, required: set[str]) -> set[str]:
    """`required` when every compile-time unit of the host build is reviewed, else nothing."""
    from nix_evidence_policy import host_compiler_graph, validate_units
    target = list(info['target']) if info['lang'] == 'rust' else ['python']
    key = (root, info.get('crate', ''), tuple(target))
    if key not in _HOST_UNITS:
        try:
            validate_units(root, host_compiler_graph(root, info.get('crate', ''), target),
                           require_compile_time=False)
            _HOST_UNITS[key] = True
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            print(f'compile-time registry does not cover {" ".join(target)}: {error}')
            _HOST_UNITS[key] = False
    return set(required) if _HOST_UNITS[key] else set()


def untracked(root: Path, paths: list[str]) -> list[str]:
    """Untracked, unignored files under `paths` (empty outside a git work tree)."""
    if not (root / '.git').exists() or not paths:
        return []
    found: list[str] = []
    for offset in range(0, len(paths), 200):
        out = subprocess.run(['git', 'ls-files', '-z', '--others', '--exclude-standard', '--',
                              *paths[offset:offset + 200]], cwd=root, capture_output=True, check=True).stdout
        found += [p for p in out.decode('utf-8').split('\0') if p]
    return sorted(set(found))


class Snapshot:
    """The inputs of one evidence record, taken before it executes.

    Pass a shared `Fingerprinter` when snapshotting many rows: each snapshot
    records only what its own row's fingerprint read.
    """

    def __init__(self, root: Path, row: dict, record: dict, fp: Fingerprinter | None = None,
                 *, nix_build: dict | None = None, allowed_outputs: dict | None = None):
        self.root = root
        self.nix_build = nix_build
        self.allowed_outputs = allowed_outputs if allowed_outputs is not None else {}
        self.nix_executed = False
        require_compile_environment(root)
        fp = fp or Fingerprinter(root, platform=False, scientific_build=record.get('kind') == 'scientific')
        with fp.tracking() as used:
            parts = fp.parts(row)
        if record.get('fingerprint') is None:
            # The record's identity: what its dependency closure was when earned.
            record = {**record, 'fingerprint': digest(canonical(parts))}
        # Reads in `#[cfg(test)]` code count only for the crate whose unit tests are built.
        test_crate = None
        if row['evidence_test'].endswith('.rs'):
            located = locate(root, row)
            if located['target'] == ['--lib']:
                test_crate = fp.crate_dir(located['crate'])
        self.source_consumers = {p.resolve() for p in used.data_reads(test_crate)}
        # Where each Rust data read is written, so a file read only by unselected items is not an input.
        self.data_sites: dict[Path, set[tuple[Path, int]]] = {}
        for scan, _ in fp._file_scans.values():
            for source, position, target in scan.sites:
                self.data_sites.setdefault(target.resolve(), set()).add((source.resolve(), position))
        env_names = {'ANTECEDENT_SCIENTIFIC_FEATURES', *parts.get('env', {}).get('hashed', {})}
        wrong = noncanonical_environment(os.environ, env_names)
        if wrong:
            raise RuntimeError('evidence is earned in the canonical environment only; unset or reset: '
                               + ', '.join(wrong))
        if os.environ.get('RUSTC_WORKSPACE_WRAPPER'):
            raise RuntimeError('workspace compiler wrappers require explicit input analysis before earning evidence')
        checker = Checker(root)
        if nix_build is None:
            version = subprocess.check_output([BUILD_ENV['RUSTC'], '-vV'], cwd=root, text=True)
            identity = dict(line.split(': ', 1) for line in version.splitlines() if ': ' in line)
            if identity.get('release') != checker.pin['release'] or identity.get('commit-hash') != checker.pin['commit']:
                raise RuntimeError('the compiler does not match tools/evidence-analyzer/pin.toml')
        else:
            tools = nix_build['tools']
            if (record.get('kind') not in ('scientific', 'calibration', 'probe')
                    or tools['rustc'] != checker.pin['release']
                    or tools['rustc_commit'] != checker.pin['commit']):
                raise RuntimeError('Nix build compiler or evidence kind is invalid')
            from nix_artifact_identity import recheck_artifact
            from nix_evidence_source import _clean_head
            self.nix_policy = _validated_build(root, nix_build)
            recheck_artifact(root, nix_build['artifact'])
            if _clean_head(root, self.allowed_outputs)[0] != nix_build['source']['commit']:
                raise RuntimeError('Nix source commit does not match the checkout')
        files = {root / rel for rel in (INPUT_DECLARATIONS, '.cargo/config.toml', '.cargo/config',
                                         'scripts/compile_time_environment.py')}
        if nix_build is not None:
            files.update(root / rel for rel in ('flake.lock', 'nix/evidence-target.nix', 'nix/evidence-deps.nix',
                         'nix/evidence-common.nix', 'nix/evidence-driver.nix', 'nix/evidence-toolchain.nix',
                         'scripts/nix_evidence_policy.py', 'scripts/nix_evidence_deps.py'))
        self.fixed_files = {path.resolve() for path in files}
        files |= used.files
        for rel in row.get('fixtures', []):
            for path in (root / rel, root / f'{rel}.json'):
                if not path.is_dir():
                    files.add(path)
        dirs = {path.relative_to(root).as_posix() for path in used.directories}
        for rel in row.get('fixtures', []):
            if (root / rel).is_dir():
                dirs.add(rel)
        info = locate(root, row)
        if nix_build is not None:
            artifact = nix_build['artifact']
            target = info['target']
            expected = ('lib', '') if target == ['--lib'] else (
                ('test', target[1]) if len(target) == 2 and target[0] == '--test' else None)
            if info['lang'] != 'rust' or expected is None or (
                    artifact['crate'], artifact['target_kind'], artifact['target_name']) != (info['crate'], *expected):
                raise RuntimeError('Nix build target does not match the evidence assertion')
        names = closure(fp.graph, info['crate']) if info['lang'] == 'rust' else list(fp.graph)
        self.crate_dirs = sorted(fp.crate_dir(name).relative_to(root).as_posix() for name in names)
        self.scope_dir = (fp.crate_dir(info['crate']).relative_to(root).as_posix()
                          if info['lang'] == 'rust' else 'python')
        self.item_tables: dict[str, bytes] = {}
        owned = {fp.crate_dir(name).resolve() for name in fp.graph}
        configs = [(root, load_toml(root / 'Cargo.toml'))]
        configs += [(fp.crate_dir(name), load_toml(fp.crate_dir(name) / 'Cargo.toml')) for name in names]
        for base, config in configs:
            custom_build = config.get('package', {}).get('build')
            if isinstance(custom_build, str) and custom_build != 'build.rs':
                raise RuntimeError('custom build-script paths require explicit input analysis')
            tables = [config.get(section, {}) for section in ('dependencies', 'dev-dependencies', 'build-dependencies', 'replace')]
            tables.append(config.get('workspace', {}).get('dependencies', {}))
            tables.extend(config.get('patch', {}).values())
            for target in config.get('target', {}).values():
                tables.extend(target.get(section, {}) for section in ('dependencies', 'dev-dependencies', 'build-dependencies'))
            for table in tables:
                for spec in table.values():
                    if isinstance(spec, dict) and 'path' in spec and (base / spec['path']).resolve() not in owned:
                        raise RuntimeError('local dependency outside the analyzed workspace must be declared as an input before earning evidence')
        for rel in ('.cargo/config', '.cargo/config.toml'):
            if (root / rel).is_file():
                config = load_toml(root / rel)
                if config.get('paths') or any(config.get('build', {}).get(key) for key in ('rustc', 'rustc-wrapper', 'rustc-workspace-wrapper')):
                    raise RuntimeError('Cargo compiler/path overrides require explicit dependency analysis before earning evidence')
        # Procedural macros and build scripts run at compile time, where coverage cannot
        # see them, and can read any file, not just their own crate's. While one runs
        # anywhere in the compilation (or the metadata saying so cannot be read), no
        # body is projected: the whole snapshot keeps complete source.
        if nix_build is None:
            compile_time = compile_time_code(root)
            if compile_time is None:
                raise RuntimeError('cannot establish compile-time dependencies; Cargo metadata is required before earning evidence')
            required = set().union(*(compile_time.get(name, {name}) for name in names))
            for name in names:
                directory = fp.crate_dir(name)
                config = load_toml(directory / 'Cargo.toml')
                if config.get('lib', {}).get('proc-macro') or (directory / 'build.rs').exists():
                    required.add(name)
        else:
            # The audited Nix compiler-unit graph supersedes the host Cargo
            # overapproximation. Keep complete source for compile-time code.
            required = {'nix-reviewed-compile-time-graph'}
        if required and nix_build is None:
            consumer = fp.crate_dir(info['crate']) if info['lang'] == 'rust' else root / 'python'
            source = (consumer / 'Cargo.toml').relative_to(root).as_posix()
            boundaries = [entry for entry in fp.declarations() if entry['source'] == source
                          and 'compile_time' in entry]
            assessed = set()
            for entry in boundaries:
                packages = entry['compile_time']
                if not isinstance(packages, list) or not all(isinstance(p, str) for p in packages):
                    raise RuntimeError(f'{INPUT_DECLARATIONS}: compile_time must list assessed package names')
                assessed.update(packages)
            # The reviewed compile-time registry is the assessed boundary for every
            # unit Cargo actually builds for this target. The metadata set above
            # over-approximates it (other platforms and features).
            assessed.update(_registry_assessed(root, info, required))
            files.update({root / 'nix/compile-time-reviewed.json', root / 'Cargo.lock'})
            if not required <= assessed:
                raise RuntimeError(f'undeclared compile-time input boundary for {source}: '
                                   f'{", ".join(sorted(required - assessed))}; add an assessed '
                                   f'compile_time input declaration to {INPUT_DECLARATIONS}')
            # A declared boundary covers data and environment reads, not just
            # source. Apply it even though no runtime path scanner reached it.
            compile_env: set[str] = set()
            if boundaries:  # extra data reads beyond the reviewed registry
                with fp.tracking() as compile_inputs:
                    _, compile_env = fp.declared([f'{source}: compile-time inputs'])
                files.update(compile_inputs.files)
                dirs.update(p.relative_to(root).as_posix() for p in compile_inputs.directories)
            for entry in boundaries:
                for pattern in entry.get('inputs', []):
                    if not any(c in pattern for c in '*?[') and not (root / pattern).is_dir():
                        files.add(root / pattern)
            env_names.update(compile_env)
            wrong = noncanonical_environment(os.environ, compile_env)
            if wrong:
                raise RuntimeError('compile-time inputs require the canonical environment; unset: ' + ', '.join(wrong))
        if nix_build is not None and self.nix_policy.get('source_projection_allowed') is not True:
            raise RuntimeError('Nix source-read audit does not permit precise per-item evidence')
        self.opaque = bool(required) if nix_build is None else False
        for name in names:
            directory = fp.crate_dir(name)
            for sub in ('src', 'tests'):
                if (directory / sub).is_dir():
                    dirs.add((directory / sub).relative_to(root).as_posix())
        for rel in ('python/antecedent', 'python/tests'):
            if (root / rel).is_dir() and row['evidence_test'].endswith('.py'):
                dirs.add(rel)
        file_rels = sorted(path.relative_to(root).as_posix() for path in files)
        stray = untracked(root, sorted(dirs) + [rel for rel in file_rels if (root / rel).exists()])
        if stray:
            raise RuntimeError('untracked files among the evidence inputs; add or ignore them: '
                               + ', '.join(stray[:10]) + (' …' if len(stray) > 10 else ''))
        inputs = [dict(kind='evidence_file', path=rel, digest=checker.input_digest('evidence_file', rel), role='external')
                  for rel in file_rels]
        for rel in sorted(dirs):
            inputs.append(dict(kind='membership', path=rel, digest=checker.input_digest('membership', rel), role='selection'))
        for declared in fp.declarations():
            for pattern in declared.get('inputs', []):
                if any(c in pattern for c in '*?[') and not any(i.get('kind') == 'glob' and i.get('path') == pattern for i in inputs):
                    inputs.append(dict(kind='glob', path=pattern, digest=checker.input_digest('glob', pattern), role='selection'))
        for name in sorted(env_names):
            inputs.append(dict(kind='env', name=name, digest=checker.input_digest('env', name), role='identity'))
        for name in BUILD_ENV:
            inputs.append(dict(kind='build_env', name=name, digest=checker.input_digest('build_env', name), role='identity'))
        self.source_trees = {rel: checker.input_digest('tree', rel) for rel in dirs if rel.endswith('/src')}
        self.manifest = dict(schema=SCHEMA, normalization=NORMALIZATION, record=record,
                             compiler={k: checker.pin[k] for k in ('release', 'commit')},
                             analyzer=ANALYZER, inputs=inputs, fallbacks=[], declarations=declarations(row))
        if nix_build is not None:
            self.manifest['nix_build'] = {
                'schema': 'nix-build-v1', 'execution': 'exact-binary',
                'source': nix_build['source'], 'artifact': nix_build['artifact'],
                'tools': nix_build['tools'], 'policy': self.nix_policy,
                'compiler_graph_sha256': digest(canonical(nix_build['compiler_graph'])),
            }

    def mark_nix_executed(self, binary: str) -> None:
        """Bind a successful exact capture to the verified Nix-built executable."""
        from nix_artifact_identity import target_binary
        if self.nix_build is None or binary != str(target_binary(self.nix_build['artifact'])):
            raise RuntimeError('captured binary does not match the Nix evidence build')
        if not self.unchanged():
            raise RuntimeError('Nix evidence inputs changed during execution')
        self.nix_executed = True

    def unchanged(self) -> bool:
        require_compile_environment(self.root)
        if self.nix_build is not None:
            from nix_artifact_identity import recheck_artifact
            from nix_evidence_source import _clean_head
            if _clean_head(self.root, self.allowed_outputs)[0] != self.nix_build['source']['commit']:
                return False
            recheck_artifact(self.root, self.nix_build['artifact'])
            if _validated_build(self.root, self.nix_build) != self.nix_policy:
                return False
        checker = Checker(self.root)
        checker.items.pending = self.item_tables
        names = next((i for i in self.manifest['inputs'] if i['kind'] == 'rust_names'), None)
        for i in self.manifest['inputs']:
            if i['kind'] == 'rust_items':
                if not checker.items.items_current(i, names):
                    return False
            elif i['kind'] == 'rust_guards':
                if not checker.items.guards_current(i):
                    return False
            elif i['kind'] == 'rust_names':
                continue
            elif checker.input_digest(i['kind'], i.get('path', i.get('name'))) != i['digest']:
                return False
        return True

    def verify_commit(self, sha: str) -> None:
        """Collected logs may attest only inputs actually present in their commit."""
        files = [i['path'] for i in self.manifest['inputs'] if i['kind'] == 'evidence_file' and (self.root / i['path']).is_file()]
        for offset in range(0, len(files), 100):
            batch = files[offset:offset + 100]
            tracked = set(subprocess.check_output(['git', 'ls-files', '-z', '--', *batch], cwd=self.root).decode().split('\0'))
            if any(path not in tracked for path in batch):
                raise RuntimeError('measurement inputs are not all tracked at the measured commit')
            changed = subprocess.run(['git', 'diff', '--quiet', sha, '--', *batch], cwd=self.root).returncode
            if changed:
                raise RuntimeError('measurement inputs differ from the measured commit')

    def publish(self) -> str:
        if self.nix_build is not None and not self.nix_executed:
            raise RuntimeError('Nix evidence cannot be published before exact execution')
        if not self.unchanged():
            raise RuntimeError('evidence inputs changed during execution; pass not recorded')
        # Share complete source-tree digests across records.
        checker = Checker(self.root)
        for guard in list(self.manifest['inputs']):
            prefix = guard.get('path', '')
            if guard['kind'] != 'membership' or not prefix.endswith('/src'):
                continue
            # Only replace files represented by the tree. Explicit reads of
            # ignored files still carry evidence even though git omits them.
            members = set(checker.members(prefix))
            inside = [i for i in self.manifest['inputs'] if i['kind'] == 'evidence_file'
                      and i.get('path') in members]
            self.manifest['inputs'] = [i for i in self.manifest['inputs'] if i not in inside]
            self.manifest['inputs'].append(dict(kind='tree', path=prefix,
                                                digest=self.source_trees[prefix], role='compile'))
        # Item tables are stored once by content; an entry names its file's table.
        from evidence_items import STORE as ITEM_TABLES
        self.outputs = [_write_addressed(self.root / ITEM_TABLES, address, data, self.root)
                        for address, data in sorted(self.item_tables.items())]
        stored, groups = grouped(self.manifest)
        for address, data in groups.items():
            self.outputs.append(_write_addressed(self.root / INPUT_GROUPS, address, data, self.root))
        ref = manifest_id(stored)
        self.outputs.append(_write_addressed(self.root / STORE, ref, canonical(stored), self.root))
        return ref


def _write_addressed(store: Path, address: str, data: bytes, root: Path) -> str:
    """Write `<address>.json` once; an existing file must hold the same bytes."""
    store.mkdir(parents=True, exist_ok=True)
    destination = store / f'{address}.json'
    if destination.exists():
        if destination.read_bytes() != data:
            raise RuntimeError('content address collision or corrupt store')
    else:
        with tempfile.NamedTemporaryFile(dir=store, delete=False) as file:
            temporary = Path(file.name)
            file.write(data)
        try:
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination.relative_to(root).as_posix()


def build_analyzer(root: Path, output: Path) -> Path | None:
    """The compiler inventory driver, or `None` (complete source is kept) without `rustc-dev`."""
    output.mkdir(parents=True, exist_ok=True)
    driver = output / 'driver'
    built = subprocess.run(['rustc', str(root / 'tools/evidence-analyzer/driver.rs'), '--edition=2021',
                            '-C', 'rpath=yes', '-o', str(driver)], cwd=root,
                           env={**os.environ, 'RUSTC_BOOTSTRAP': '1'}, capture_output=True, text=True)
    if built.returncode != 0:
        print('scientific records: compiler inventory unavailable (rustup component add rustc-dev); '
              'manifests keep complete source', flush=True)
        return None
    return driver


# Per-process caches for earning many rows of one build. Each key carries everything
# its value depends on, so a cached value is the value that would be recomputed.
_INVENTORIES: dict[tuple, dict] = {}
_POLICIES: dict[tuple, dict] = {}


def _validated_build(root: Path, nix_build: dict) -> dict:
    """`validate_build`, once per unchanged daemon configuration, audit and unit graph."""
    from evidence_manifests import _stat_signature
    from nix_evidence_policy import NIX_OVERRIDES, SYSTEM_CONFIG, validate_build
    key = (str(root), _stat_signature(SYSTEM_CONFIG), tuple(sorted(NIX_OVERRIDES.intersection(os.environ))),
           _stat_signature(root / 'nix/compile-time-reviewed.json'), _stat_signature(root / 'Cargo.lock'),
           digest(canonical(nix_build['compiler_graph'])))
    if key not in _POLICIES:
        _POLICIES[key] = validate_build(root, nix_build)
    return _POLICIES[key]


def publish_executed(snapshot: Snapshot, built: dict, coverage: dict[str, set[tuple[int, int]]]) -> str:
    """Publish an evidence record that ran the Nix-built executable of `built`: bind it to
    that executable, keep only the items its run executed and what they depend on
    (`coverage`), and store its manifest. Shared by scientific assertions and calibration
    measurements; returns the manifest's address."""
    from nix_artifact_identity import target_binary
    snapshot.mark_nix_executed(str(target_binary(built["artifact"])))
    select_items(snapshot, Path(built["artifact"]["output"]) / "compiler-inventory", coverage,
                 source_root=Path(built["source"]["store_path"]))
    return snapshot.publish()


def select_items(snapshot: Snapshot, directory: Path, coverage: dict[str, set[tuple[int, int]]],
                 *, source_root: Path | None = None) -> None:
    """Replace a Rust assertion's whole sources with the items it depends on.

    `coverage` maps checkout paths to executed (line, column) positions and
    `directory` holds the compiler inventory of the same compilation. Compile-time
    code that may read any source (host builds only), a missing inventory or
    missing coverage keep complete source; so do files read as data, files the
    parser cannot read, and the files of an assertion that observes positions.
    """
    from evidence_items import Selection, load_inventory
    # A Nix capture was checked unchanged when it was marked executed, with nothing run
    # since; publication checks again after selecting.
    if not snapshot.nix_executed and not snapshot.unchanged():
        raise RuntimeError('source changed during instrumented execution')
    if snapshot.opaque:
        snapshot.manifest['fallbacks'] = [
            dict(path=i['path'], reason='compile-time code may read any source; source retained')
            for i in snapshot.manifest['inputs'] if i['kind'] == 'membership']
        return
    inventory = load_inventory(directory)
    if not inventory.items or not coverage:
        if source_root is not None:
            raise RuntimeError('Nix precision requires compiler inventory and instrumented coverage')
        return
    root = snapshot.root.resolve()
    selection = Selection(root, inventory)
    for path, positions in sorted(coverage.items()):
        try:
            rel = Path(path).resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        if not rel.endswith('.rs'):
            continue
        file = selection.file(rel)
        if file is None:
            continue
        for line, column in sorted(positions):
            selection.execute(rel, file.offset(line, column))
    for path in snapshot.source_consumers:
        if path.suffix == '.rs' and path.is_relative_to(root):
            selection.keep_whole(path.relative_to(root).as_posix(), 'read as data')
    entries, tables = selection.entries(Checker(root), snapshot.crate_dirs, snapshot.scope_dir)

    def selected(source: Path, position: int) -> bool:
        """Whether a data read is written inside an item the assertion depends on."""
        if not source.is_relative_to(root):
            return True
        rel = source.relative_to(root).as_posix()
        if rel in selection.whole:
            return True
        file = selection.file(rel) if rel in selection.parts else None
        if file is None or '\r' in file.text:
            return file is not None
        for locator, parts in selection.parts[rel].items():
            node = file.by_locator.get(locator)
            if node is not None and node.start <= position < (node.end if 'full' in parts else node.head_end):
                return True
        return False

    def unread(entry: dict) -> bool:
        if entry['kind'] != 'evidence_file':
            return False
        target = (root / entry['path']).resolve()
        if (target in snapshot.fixed_files or target.name == 'Cargo.toml'
                or target.suffix == '.rs' or target not in snapshot.source_consumers):
            return False
        sites = snapshot.data_sites.get(target)
        return bool(sites) and not any(selected(source, position) for source, position in sites)

    def item_tracked(rel: str) -> bool:
        """A crate source the item entries stand for; build scripts stay whole files."""
        return rel.endswith('.rs') and any(rel.startswith(f'{d}/src/') or rel.startswith(f'{d}/tests/')
                                           for d in snapshot.crate_dirs)

    rust_trees = {f'{d}/{sub}' for d in snapshot.crate_dirs for sub in ('src', 'tests')}
    kept = [i for i in snapshot.manifest['inputs']
            if not (i['kind'] == 'evidence_file' and item_tracked(i['path']))
            and not (i['kind'] == 'membership' and i['path'] in rust_trees)
            and not unread(i)]
    seen = {(i['kind'], i.get('path', i.get('name'))) for i in kept}
    kept.extend(e for e in entries if (e['kind'], e['path']) not in seen)
    snapshot.manifest['inputs'] = kept
    snapshot.manifest['whole_sources'] = selection.reasons
    snapshot.item_tables = tables
