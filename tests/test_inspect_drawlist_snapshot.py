#!/usr/bin/env python3
"""The inventory tool never claims that an outer-valid blob was GL-restored."""
import hashlib
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("snapshot_inspector", ROOT / "tools/inspect_drawlist_snapshot.py")
INSPECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSPECTOR)


def u64(value):
    return struct.pack("<Q", value)


def blob(value):
    return u64(len(value)) + value


def archive(records=None, version=1, context=b"opaque-context"):
    records = [(9, 2)] if records is None else records
    payload = blob(b"a" * 64) + u64(36) + u64(12) + u64(256)
    payload += b"".join(blob(text) for text in (b"vendor", b"renderer", b"version", context))
    payload += u64(0) * 25 + u64(len(records))
    for identity, namespace in records:
        payload += u64(identity) + u64(namespace) + blob(b"signature") + u64(0) + blob(b"opaque-resource")
    magic = {1: b"RDGLSN01", 2: b"RDGLSN02"}[version]
    return magic + u64(len(payload)) + hashlib.sha256(payload).digest() + payload


class SnapshotInspectorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def inspect(self, data):
        path = self.root / "state.bin"
        path.write_bytes(data)
        return INSPECTOR.inspect(path)

    def test_inventory_and_explicit_non_restore_claim(self):
        data = archive()
        result = self.inspect(data)
        self.assertEqual(result["archive_sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(result["resource_count"], 1)
        self.assertEqual(result["resources"][0]["id"], 9)
        self.assertEqual(result["snapshot_api_version"], 1)
        self.assertIn("not a restoration receipt", result["verification"])

    def test_v2_keeps_distinct_archive_version_and_outer_only_contract(self):
        result = self.inspect(archive(version=2))
        self.assertEqual(result["snapshot_api_version"], 2)
        self.assertEqual(result["resource_count"], 1)
        self.assertIn("not a restoration receipt", result["verification"])

    def test_unknown_archive_version_is_refused(self):
        data = bytearray(archive())
        data[7] = ord("3")
        with self.assertRaisesRegex(ValueError, "magic/version"):
            self.inspect(data)

    def test_versioned_outer_context_limit(self):
        for version, limit in ((1, 65536), (2, 65536 + 48)):
            with self.subTest(version=version):
                self.assertEqual(self.inspect(archive(version=version, context=b"x" * limit))
                                 ["snapshot_api_version"], version)
                with self.assertRaisesRegex(ValueError, "exceeds limit"):
                    self.inspect(archive(version=version, context=b"x" * (limit + 1)))
        with self.assertRaisesRegex(ValueError, "exceeds limit"):
            self.inspect(archive(version=1, context=b"x" * (65536 + 48)))

    def test_truncated_outer_envelope(self):
        for count in (0, 7, 20, 47, 48, 100):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.inspect(archive()[:count])

    def test_corrupt_payload(self):
        data = bytearray(archive())
        data[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.inspect(data)

    def test_duplicate_identity(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.inspect(archive([(9, 2), (9, 2)]))

    def test_unknown_namespace_width(self):
        with self.assertRaisesRegex(ValueError, "identity"):
            self.inspect(archive([(9, 1 << 32)]))

    def test_symlink_refused(self):
        target = self.root / "actual.bin"
        target.write_bytes(archive())
        alias = self.root / "alias.bin"
        alias.symlink_to(target)
        with self.assertRaises(OSError):
            INSPECTOR.inspect(alias)

    def test_reader_length_limits(self):
        reader = INSPECTOR.Reader(u64(100) + b"tiny")
        with self.assertRaisesRegex(ValueError, "exceeds limit"):
            reader.blob(16)
        reader = INSPECTOR.Reader(u64(100) + b"tiny")
        with self.assertRaisesRegex(ValueError, "truncated"):
            reader.blob(100)


if __name__ == "__main__":
    unittest.main()
