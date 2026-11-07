"""Inspect a Nix-built Antecedent wheel without authorizing evidence.

The caller must independently establish the Git/source mapping, builder policy,
test runtime dependencies and execution binding. No tests or records are run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tomllib
import zipfile

from nix_artifact_identity import ArtifactIdentityError, _file_hash, _nix, _store_path
from nix_compile_time_inventory import compiler_units


def _wheel_contents(wheel: Path, source: Path) -> tuple[str, str]:
    """Require Python package source to match the imported committed tree."""
    package = source / 'python/antecedent'
    if package.is_symlink() or not package.is_dir():
        raise ArtifactIdentityError('wheel source package is missing or linked')
    expected = {}
    for path in sorted(package.rglob('*')):
        if path.is_symlink():
            raise ArtifactIdentityError('wheel source package contains a symlink')
        if path.is_file():
            expected['antecedent/' + path.relative_to(package).as_posix()] = path
    if not expected:
        raise ArtifactIdentityError('wheel source package is empty')
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ArtifactIdentityError('wheel contains duplicate entries')
        for name in names:
            pure = PurePosixPath(name)
            if pure.is_absolute() or '\\' in name or '..' in pure.parts:
                raise ArtifactIdentityError('wheel contains an unsafe path')
        actual = {name for name in names if name.startswith('antecedent/') and not name.endswith('/')}
        extensions = sorted(name for name in actual if re.fullmatch(r'antecedent/_native(?:\.[^/]+)?\.so', name))
        if len(extensions) != 1 or actual - set(extensions) != set(expected):
            raise ArtifactIdentityError('wheel package membership differs from committed source')
        for name, path in expected.items():
            if archive.read(name) != path.read_bytes():
                raise ArtifactIdentityError(f'wheel package source differs: {name}')
        binary = archive.read(extensions[0])
    return extensions[0], hashlib.sha256(binary).hexdigest()


def inspect_python_wheel(root: Path, output: str, source: str, *, store: Path = Path('/nix/store')) -> dict:
    """Re-evaluate and identify a wheel and its interpreter; never attest execution."""
    root, store = root.resolve(), store.resolve()
    output, source = _store_path(output, store), _store_path(source, store)
    for item in (output, source):
        if Path(item).is_symlink() or not Path(item).is_dir():
            raise ArtifactIdentityError('missing or linked Nix source/output')
    recipe_paths = [root / 'nix/evidence-python-wheel.nix', root / 'nix/evidence-toolchain.nix', root / 'flake.lock']
    recipe_hashes = {str(path.relative_to(root)): _file_hash(path) for path in recipe_paths}
    drv = _nix('nix-instantiate', '--extra-experimental-features', 'nix-command flakes', str(recipe_paths[0]),
               '--arg', 'src', 'builtins.storePath ' + json.dumps(source))
    _store_path(drv, store, derivation=True)
    try:
        definition = json.loads(_nix('nix', '--extra-experimental-features', 'nix-command', 'derivation', 'show', drv))[drv]
        metadata = json.loads(_nix('nix', '--extra-experimental-features', 'nix-command', 'path-info', '--json', output))[output]
        if definition['outputs']['out']['path'] != output or source not in definition['inputSrcs']:
            raise ArtifactIdentityError('Python derivation does not match source/output')
        if metadata['deriver'] != drv or source not in metadata['references']:
            raise ArtifactIdentityError('Python output metadata does not match source/deriver')
        output_nar = metadata['narHash']
        if _nix('nix', '--extra-experimental-features', 'nix-command', 'hash', 'path', output) != output_nar:
            raise ArtifactIdentityError('Python output differs from store metadata')
        source_nar = _nix('nix', '--extra-experimental-features', 'nix-command', 'hash', 'path', source)
        sri = r'sha256-[A-Za-z0-9+/]{43}='
        if not re.fullmatch(sri, output_nar) or not re.fullmatch(sri, source_nar):
            raise ArtifactIdentityError('invalid Python Nix NAR hash')
        out = Path(output)
        if (out / 'source-root').is_symlink() or (out / 'source-root').read_text().strip() != source:
            raise ArtifactIdentityError('Python source root differs')
        wheels = list((out / 'wheels').glob('*.whl'))
        if len(wheels) != 1:
            raise ArtifactIdentityError('expected exactly one Python wheel')
        wheel_sha = _file_hash(wheels[0])
        extension_name, extension_sha = _wheel_contents(wheels[0], Path(source))
        config = json.loads((out / 'python-config.json').read_text())
        python = Path(config['executable'])
        if (config.get('implementation') != 'cpython' or not config.get('version', '').startswith('3.13.')
                or not str(python).startswith(str(store) + '/')
                or (out / 'bin/python3.13').resolve() != python.resolve()):
            raise ArtifactIdentityError('Python interpreter identity is not pinned to the output')
        versions = (out / 'rustc-version').read_text()
        fields = dict(line.split(': ', 1) for line in versions.splitlines() if ': ' in line)
        pin = tomllib.loads((root / 'tools/evidence-analyzer/pin.toml').read_text())
        if fields.get('release') != pin['release'] or fields.get('commit-hash') != pin['commit']:
            raise ArtifactIdentityError('Python wheel compiler differs from evidence pin')
        cargo = (out / 'cargo-version').read_text().strip()
        maturin = (out / 'maturin-version').read_text().strip()
        if not re.fullmatch(r'cargo \d+\.\d+\.\d+(?: \([^\n]*\))?', cargo) or not re.fullmatch(r'maturin \d+\.\d+\.\d+[^\n]*', maturin):
            raise ArtifactIdentityError('missing or invalid Python wheel build tools')
        cargo_metadata = json.loads((out / 'cargo-metadata.json').read_text())
        cargo_graph = json.loads((out / 'cargo-unit-graph.json').read_text())
        graph = compiler_units(cargo_metadata, cargo_graph)
        if len(graph['roots']) != 1:
            raise ArtifactIdentityError('Python wheel unit graph has multiple roots')
        unit = graph['units'][graph['roots'][0]]
        if unit['name'] != 'antecedent-py' or unit['mode'] != 'build' or 'cdylib' not in unit['target_kind']:
            raise ArtifactIdentityError('Python wheel graph does not describe the extension build')
    except ArtifactIdentityError:
        raise
    except (OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile) as error:
        raise ArtifactIdentityError('invalid Python wheel build identity') from error
    if any(_file_hash(path) != recipe_hashes[str(path.relative_to(root))] for path in recipe_paths):
        raise ArtifactIdentityError('Python wheel recipe changed during inspection')
    return {'schema': 1, 'derivation': drv, 'source': source, 'source_nar_hash': source_nar,
            'output': output, 'output_nar_hash': output_nar, 'wheel_filename': wheels[0].name,
            'wheel_sha256': wheel_sha, 'extension_filename': extension_name,
            'extension_sha256': extension_sha, 'python': config, 'cargo': cargo, 'maturin': maturin,
            'rustc': fields['release'], 'rustc_commit': fields['commit-hash'], 'recipe_sha256': recipe_hashes,
            'compiler_graph': graph, 'runtime_requirements_verified': False,
            'sandbox_verified': False, 'evidence_authorized': False}
