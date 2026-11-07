"""Inspect a locked Nix Python runtime without issuing an evidence record."""
from __future__ import annotations

from email.parser import Parser
import json
from pathlib import Path
import re
import tomllib
from urllib.parse import unquote

from nix_artifact_identity import ArtifactIdentityError, _file_hash, _nix, _store_path
from nix_python_evidence_identity import inspect_python_wheel


def _canonical_name(name: str) -> str:
    return re.sub(r'[-_.]+', '-', name).lower()


def inspect_python_runtime(root: Path, output: str, source: str, wheel_output: str,
                           *, store: Path = Path('/nix/store')) -> dict:
    """Verify lock, installed package versions, source and native extension.

    This checks identity only: it does not assert daemon policy, that a test
    used this interpreter, or authorize publication of any evidence.
    """
    root, store = root.resolve(), store.resolve()
    output = _store_path(output, store)
    source = _store_path(source, store)
    wheel_output = _store_path(wheel_output, store)
    for item in (output, source, wheel_output):
        if Path(item).is_symlink() or not Path(item).is_dir():
            raise ArtifactIdentityError('missing or linked Python runtime input')
    wheel = inspect_python_wheel(root, wheel_output, source, store=store)
    recipe = root / 'nix/evidence-python-runtime.nix'
    recipe_hash = _file_hash(recipe)
    drv = _nix('nix-instantiate', '--extra-experimental-features', 'nix-command flakes', str(recipe),
               '--arg', 'src', 'builtins.storePath ' + json.dumps(source),
               '--arg', 'wheelOutput', 'builtins.storePath ' + json.dumps(wheel_output))
    _store_path(drv, store, derivation=True)
    try:
        definition = json.loads(_nix('nix', '--extra-experimental-features', 'nix-command', 'derivation', 'show', drv))[drv]
        metadata = json.loads(_nix('nix', '--extra-experimental-features', 'nix-command', 'path-info', '--json', output))[output]
        if (definition['outputs']['out']['path'] != output
                or source not in definition['inputSrcs']):
            raise ArtifactIdentityError('Python runtime derivation differs from request')
        if (metadata['deriver'] != drv or source not in metadata['references']
                or wheel_output not in metadata['references']):
            raise ArtifactIdentityError('Python runtime store references differ from request')
        nar = metadata['narHash']
        if (not re.fullmatch(r'sha256-[A-Za-z0-9+/]{43}=', nar)
                or _nix('nix', '--extra-experimental-features', 'nix-command', 'hash', 'path', output) != nar):
            raise ArtifactIdentityError('Python runtime differs from store metadata')
        out = Path(output)
        for name, expected in (('source-root', source), ('wheel-output', wheel_output)):
            path = out / name
            if path.is_symlink() or not path.is_file() or path.read_text(encoding="utf-8").strip() != expected:
                raise ArtifactIdentityError('Python runtime source or wheel identity differs')
        interpreter = out / 'bin/python3.13'
        if not interpreter.is_symlink() or interpreter.resolve() != Path(wheel['python']['executable']).resolve():
            raise ArtifactIdentityError('Python runtime interpreter differs from the wheel interpreter')
        lock_path = Path(source) / 'python/uv.lock'
        if lock_path.is_symlink() or not lock_path.is_file():
            raise ArtifactIdentityError('Python runtime dependency lock is missing or linked')
        locked = tomllib.loads(lock_path.read_text(encoding='utf-8'))['package']
        wanted = {_canonical_name(p['name']): p for p in locked if p['name'] != 'antecedent'}
        if len(wanted) != sum(p['name'] != 'antecedent' for p in locked):
            raise ArtifactIdentityError('Python runtime lock has ambiguous package names')
        package_file = out / 'locked-packages.json'
        if package_file.is_symlink() or not package_file.is_file():
            raise ArtifactIdentityError('Python runtime package report is missing or linked')
        selected = json.loads(package_file.read_text(encoding='utf-8'))
        names = [_canonical_name(row['name']) for row in selected]
        if len(names) != len(set(names)) or set(names) != set(wanted):
            raise ArtifactIdentityError('Python runtime dependency membership differs from lock')
        for row in selected:
            package = wanted[_canonical_name(row['name'])]
            matches = [w for w in package.get('wheels', [])
                       if unquote(w['url'].rsplit('/', 1)[-1]) == row['file']
                       and w['hash'] == 'sha256:' + row['sha']]
            if row['version'] != package['version'] or len(matches) != 1:
                raise ArtifactIdentityError('Python runtime wheel differs from lock')
        site = out / 'site'
        if site.is_symlink() or not site.is_dir():
            raise ArtifactIdentityError('Python runtime site is missing or linked')
        installed = {}
        for path in site.glob('*.dist-info/METADATA'):
            if path.is_symlink() or not path.is_file():
                raise ArtifactIdentityError('Python runtime metadata is missing or linked')
            info = Parser().parsestr(path.read_text(encoding='utf-8'))
            name = _canonical_name(info['Name'])
            if name in installed:
                raise ArtifactIdentityError('Python runtime has duplicate distributions')
            installed[name] = info['Version']
        project = tomllib.loads((Path(source) / 'python/pyproject.toml').read_text(encoding='utf-8'))['project']
        expected_versions = {name: package['version'] for name, package in wanted.items()}
        expected_versions[_canonical_name(project['name'])] = project['version']
        if installed != expected_versions:
            raise ArtifactIdentityError('Python runtime installed distributions differ from lock')
        native = site / wheel['extension_filename']
        if _file_hash(native) != wheel['extension_sha256']:
            raise ArtifactIdentityError('Python runtime native extension differs from wheel')
        source_package = Path(source) / 'python/antecedent'
        for path in source_package.rglob('*'):
            if path.is_file():
                if path.is_symlink():
                    raise ArtifactIdentityError('Python source package contains a link')
                installed_path = site / 'antecedent' / path.relative_to(source_package)
                if installed_path.is_symlink() or not installed_path.is_file() or installed_path.read_bytes() != path.read_bytes():
                    raise ArtifactIdentityError('Python runtime package source differs from wheel')
    except ArtifactIdentityError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ArtifactIdentityError('invalid Python runtime identity') from error
    if _file_hash(recipe) != recipe_hash:
        raise ArtifactIdentityError('Python runtime recipe changed during inspection')
    return {'schema': 1, 'derivation': drv, 'source': source, 'source_nar_hash': wheel['source_nar_hash'],
            'wheel_output': wheel_output, 'wheel_sha256': wheel['wheel_sha256'],
            'output': output, 'output_nar_hash': nar, 'runtime_recipe_sha256': recipe_hash,
            'interpreter': str(interpreter.resolve()), 'locked_packages': selected,
            'locked_runtime_metadata_verified': True,
            'sandbox_verified': False, 'evidence_authorized': False}
