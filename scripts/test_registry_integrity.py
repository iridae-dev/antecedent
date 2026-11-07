"""Package verification checks; no scientific or calibration work executes."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from registry_integrity import RegistryIntegrityError, verify_package, verify_registry


class RegistryIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.directory = root / 'sample-1.0.0'
        self.directory.mkdir()
        (self.directory / 'src').mkdir()
        (self.directory / 'src/lib.rs').write_bytes(b'contents')
        self.archive = root / 'sample-1.0.0.crate'
        with tarfile.open(self.archive, 'w:gz') as tar:
            data = b'contents'
            info = tarfile.TarInfo('sample-1.0.0/src/lib.rs')
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        self.checksum = hashlib.sha256(self.archive.read_bytes()).hexdigest()

    def verify(self):
        verify_package(self.directory, self.archive, self.checksum)

    def test_exact_package_and_cargo_metadata(self):
        (self.directory / '.cargo-ok').write_text('{"v":1}')
        (self.directory / '.cargo-checksum.json').write_text(json.dumps({
            'package': self.checksum,
            'files': {'src/lib.rs': hashlib.sha256(b'contents').hexdigest()},
        }))
        self.verify()

    def test_tampered_archive_is_rejected(self):
        self.archive.write_bytes(self.archive.read_bytes() + b'x')
        with self.assertRaisesRegex(RegistryIntegrityError, 'archive checksum'):
            self.verify()

    def test_modified_missing_and_extra_source_are_rejected(self):
        f = self.directory / 'src/lib.rs'
        f.write_bytes(b'changed!')
        with self.assertRaisesRegex(RegistryIntegrityError, 'file mismatch'):
            self.verify()
        f.unlink()
        with self.assertRaisesRegex(RegistryIntegrityError, 'membership mismatch'):
            self.verify()
        f.write_bytes(b'contents')
        (self.directory / 'extra').write_text('x')
        with self.assertRaisesRegex(RegistryIntegrityError, 'membership mismatch'):
            self.verify()

    def test_linked_source_is_rejected(self):
        f = self.directory / 'src/lib.rs'
        f.unlink()
        f.symlink_to(self.archive)
        with self.assertRaisesRegex(RegistryIntegrityError, 'unsupported package entry'):
            self.verify()

    def test_unsafe_archive_member_is_rejected(self):
        with tarfile.open(self.archive, 'w:gz') as tar:
            info = tarfile.TarInfo('sample-1.0.0/../escape')
            info.size = 1
            tar.addfile(info, io.BytesIO(b'x'))
        self.checksum = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        with self.assertRaisesRegex(RegistryIntegrityError, 'unsafe archive'):
            self.verify()

    def test_registry_metadata_is_bound_to_exact_locked_package(self):
        root = Path(self.tmp.name)
        src = root / 'registry/src/index/sample-1.0.0'
        cache = root / 'registry/cache/index'
        src.parent.mkdir(parents=True)
        cache.mkdir(parents=True)
        self.directory.rename(src)
        self.archive.rename(cache / self.archive.name)
        (root / 'Cargo.lock').write_text(
            '[[package]]\nname="sample"\nversion="1.0.0"\n'
            'source="registry+test"\nchecksum="' + self.checksum + '"\n'
        )
        metadata = {'packages': [{'name': 'sample', 'version': '1.0.0',
                    'source': 'registry+test', 'manifest_path': str(src / 'Cargo.toml')}]}
        verify_registry(root, metadata)
        metadata['packages'][0]['source'] = 'registry+other'
        with self.assertRaisesRegex(RegistryIntegrityError, 'missing locked'):
            verify_registry(root, metadata)


if __name__ == '__main__':
    unittest.main()
