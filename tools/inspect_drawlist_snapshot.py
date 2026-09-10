#!/usr/bin/env python3
"""Read-only inventory/digest inspector for RDGLSN01/RDGLSN02 checkpoints.

This validates the outer envelope, not the GL codec's semantic compatibility.
Only the snapshot extension's bounded preflight and complete live read-back
verification can establish successful restoration. Program blob hashes include
process-local uniform locations and are not semantic program comparisons.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import struct

MAX_ARCHIVE = 2 * 1024**3
MAX_BLOB = 256 * 1024**2


class Reader:
    def __init__(self, data: bytes | memoryview):
        self.data, self.at = memoryview(data), 0

    def raw(self, count: int) -> memoryview:
        if count < 0 or count > len(self.data) - self.at:
            raise ValueError("truncated snapshot field")
        result = self.data[self.at:self.at + count]
        self.at += count
        return result

    def u64(self) -> int:
        return struct.unpack("<Q", self.raw(8))[0]

    def blob(self, limit: int = MAX_BLOB) -> memoryview:
        count = self.u64()
        if count > limit:
            raise ValueError("snapshot blob exceeds limit")
        return self.raw(count)

    def text(self, limit: int = 4096) -> str:
        return bytes(self.blob(limit)).decode("utf-8", errors="strict")


def digest(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def inspect(path: Path) -> dict:
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
    with os.fdopen(os.open(path, flags), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 48 <= info.st_size <= MAX_ARCHIVE:
            raise ValueError("snapshot must be a bounded regular file")
        raw = stream.read(info.st_size + 1)
        if len(raw) != info.st_size:
            raise ValueError("snapshot size changed while reading")
    versions = {b"RDGLSN01": 1, b"RDGLSN02": 2}
    if raw[:8] not in versions:
        raise ValueError("snapshot magic/version mismatch")
    payload = memoryview(raw)[48:]
    if struct.unpack_from("<Q", raw, 8)[0] != len(payload):
        raise ValueError("snapshot envelope length mismatch")
    if hashlib.sha256(payload).digest() != raw[16:48]:
        raise ValueError("snapshot payload checksum mismatch")
    reader = Reader(payload)
    capture = reader.text(64)
    if len(capture) != 64 or any(c not in "0123456789abcdef" for c in capture):
        raise ValueError("invalid capture SHA-256")
    result = {"schema": "pvrgpu.drawlist-snapshot-inventory.v1", "capture_sha256": capture,
              "snapshot_api_version": versions[raw[:8]],
              "section_version": reader.u64(), "event": reader.u64(),
              "next_chunk_offset": reader.u64(), "vendor": reader.text(),
              "renderer": reader.text(), "version": reader.text()}
    # V2 wraps the same bounded native GL state in 48 bytes of typed context
    # metadata (renderbuffer selector and optional pack-direction state).
    context = reader.blob(65536 + (48 if result["snapshot_api_version"] == 2 else 0))
    result["context_state_sha256"] = digest(context)
    result["pack"] = [reader.u64() for _ in range(12)]
    result["unpack"] = [reader.u64() for _ in range(12)]
    result["default_fbo_id"] = reader.u64()
    count = reader.u64()
    if count > 16384:
        raise ValueError("snapshot resource count exceeds limit")
    resources, seen = [], set()
    for _ in range(count):
        resource_id, namespace = reader.u64(), reader.u64()
        if not resource_id or resource_id in seen or namespace > 0xffffffff:
            raise ValueError("invalid/duplicate snapshot resource identity")
        seen.add(resource_id)
        signature = reader.blob(8 * 1024**2)
        locations = reader.u64()
        if locations > 65536:
            raise ValueError("snapshot program location count exceeds limit")
        mapping = [[reader.u64(), reader.u64()] for _ in range(locations)]
        if any(a > 0xffffffff or b > 0xffffffff for a, b in mapping):
            raise ValueError("invalid uniform location width")
        blob = reader.blob()
        resources.append({"id": resource_id, "namespace": namespace,
                          "signature_bytes": len(signature), "signature_sha256": digest(signature),
                          "locations": mapping, "blob_bytes": len(blob), "blob_sha256": digest(blob)})
    if reader.at != len(payload):
        raise ValueError("snapshot has trailing payload bytes")
    result.update(resources=resources, resource_count=len(resources), archive_bytes=len(raw),
                  archive_sha256=digest(raw), payload_sha256=digest(payload),
                  verification="outer-envelope-only; not a restoration receipt")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="snapshot/state.bin; read-only")
    parser.add_argument("--resource", type=int, help="print only the selected original ResourceId")
    args = parser.parse_args()
    try:
        result = inspect(args.archive)
        if args.resource is not None:
            result["resources"] = [r for r in result["resources"] if r["id"] == args.resource]
            if not result["resources"]:
                raise ValueError("requested resource is absent")
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, UnicodeError) as error:
        parser.exit(1, f"inspect_drawlist_snapshot.py: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
