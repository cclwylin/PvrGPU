#!/usr/bin/env python3
"""Checksum-valid v2 snapshot rejection tests, not image or rollback tests.

Ordinary unittest discovery runs synthetic, GPU-free mutation/guard tests only.
Real integration is opt-in and always uses the official wrapper in a NEW process:

  python3 tests/test_drawlist_snapshot_rejection.py run \
    --snapshot-dir /absolute/verified/snapshot --capture /absolute/input.rdc \
    --player /absolute/player --renderdoc-lib /absolute/librenderdoc.dylib \
    --mesa-prefix /absolute/mesa --backend llvmpipe --gles-version 3.1 \
    --through-draw 2 --outdir /absolute/NEW-output \
    --codec-source /absolute/frozen/gl_replay_snapshot.inl \
    --safety-source /absolute/frozen/gl_snapshot_codec_safety.inl

For native replay also supply --bridge. The source checkpoint must already have
an independently verified successful v2 Save/Load. This runner does not certify
that positive run, build anything, or infer that no receipt means no GL writes.
It preserves the original checkpoint and reseals each MUTATED copy's envelope,
manifest state identity and embedded receipt checksum. Those copied receipts
are intentionally fabricated test inputs, NEVER successful restore evidence.

Wire offsets below derive from the locked RenderDoc v2 codec and its original
FramebufferInitialData serializer, not searches for ResourceId byte patterns.
Unknown layouts fail closed. A tag5 -> existing texture ID mutation is negative
only because v2 explicitly stores the expected attachment namespace tag.

Use --depth24-mode for the separate single-mip D24 texture fixture. It generates
two canonical-low-byte negatives instead of changing the default 25 D16/FBO
cases. Both modes retain the same new-process wrapper and resealing contract.
Use --buffer-target-mode for three separate buffer signature negatives: invalid
last-binding target, changed physical size, and changed cached metadata size.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MAX_ARCHIVE = 2 * 1024**3
MAX_BLOB = 256 * 1024**2
MAGIC = b"RDGLSN02"
STORAGE_ERROR = "snapshot boundary storage/program signature differs from fresh controller (v1 unsupported)"
D16_ID_ERROR = "snapshot D16 blob embedded identity/codec mismatch"
D16_SIGNATURE_ERROR = "snapshot D16 blob storage signature mismatch"
FBO_TYPE_ERROR = "snapshot framebuffer attachment typed namespace mismatch"
FBO_SUBRESOURCE_ERROR = "snapshot D16 framebuffer attachment has invalid subresource/type metadata"
D24_CANONICAL_ERROR = "snapshot D24 raw data is not canonical normalized U32"
BUFFER_TARGET_ERROR = "snapshot buffer last-binding target is invalid"
BUFFER_TARGETS = frozenset((0, 0x8892, 0x92c0, 0x8f36, 0x8f37, 0x8f3f, 0x90ee,
                           0x8893, 0x88eb, 0x88ec, 0x9192, 0x90d2, 0x8c2a,
                           0x8c8e, 0x8a11, 0x80ee))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def u64(value):
    return struct.pack("<Q", value)


def blob(data):
    return u64(len(data)) + data


def set_word(data, offset, value, width=8):
    require(width in (1, 4, 8) and 0 <= offset <= len(data) - width, "mutation span")
    changed = bytearray(data)
    changed[offset:offset + width] = value.to_bytes(width, "little")
    require(changed != data, "mutation must change the selected field")
    return bytes(changed)


class Cursor:
    def __init__(self, data):
        self.data, self.at = data, 0

    def take(self, size):
        require(0 <= size <= len(self.data) - self.at, "truncated field")
        result = self.data[self.at:self.at + size]
        self.at += size
        return result

    def integer(self, width=8):
        return int.from_bytes(self.take(width), "little")

    def blob(self, limit=MAX_BLOB):
        size = self.integer()
        require(size <= limit, "blob limit")
        return self.take(size)

    def done(self):
        require(self.at == len(self.data), "trailing bytes")


@dataclass(frozen=True)
class Record:
    identity: int
    namespace: int
    signature: bytes
    locations: tuple
    data: bytes

    def encode(self):
        return (u64(self.identity) + u64(self.namespace) + blob(self.signature) +
                u64(len(self.locations)) + b"".join(u64(a) + u64(b) for a, b in self.locations) +
                blob(self.data))


@dataclass(frozen=True)
class Archive:
    prefix: bytes  # exact metadata/context bytes through default FBO, excluding count
    section: int
    event: int
    capture_sha: str
    records: tuple

    def encode(self):
        payload = self.prefix + u64(len(self.records)) + b"".join(r.encode() for r in self.records)
        require(len(payload) <= MAX_ARCHIVE - 48, "archive limit")
        return MAGIC + u64(len(payload)) + hashlib.sha256(payload).digest() + payload

    @classmethod
    def decode(cls, data, *, unique=True):
        require(48 <= len(data) <= MAX_ARCHIVE and data[:8] == MAGIC, "v2 archive magic/size")
        require(int.from_bytes(data[8:16], "little") == len(data) - 48, "payload size")
        require(hashlib.sha256(data[48:]).digest() == data[16:48], "payload checksum")
        c = Cursor(data[48:])
        capture = c.blob(64).decode("ascii")
        require(len(capture) == 64 and all(x in "0123456789abcdef" for x in capture), "capture digest")
        section, event = c.integer(), c.integer()
        c.integer()  # next chunk offset, retained verbatim
        for _ in range(3):
            c.blob(4096)
        c.blob(65536 + 48)
        for _ in range(25):  # pack/unpack + default FBO
            c.integer()
        prefix = c.data[:c.at]
        count = c.integer()
        require(0 < count <= 16384, "resource count")
        records, seen = [], set()
        for _ in range(count):
            identity, namespace = c.integer(), c.integer()
            require(identity and namespace <= 0xffffffff, "resource identity/namespace")
            require(not unique or identity not in seen, "duplicate identity")
            seen.add(identity)
            signature = c.blob(8 * 1024**2)
            n = c.integer()
            require(n <= 65536 and (namespace == 9 or n == 0), "location count")
            locations = tuple((c.integer(), c.integer()) for _ in range(n))
            require(all(a <= 0xffffffff and b <= 0xffffffff for a, b in locations), "location width")
            records.append(Record(identity, namespace, signature, locations, c.blob()))
        c.done()
        result = cls(prefix, section, event, capture, tuple(records))
        require(result.encode() == data, "archive re-encoding changed bytes")
        return result


def d16_fields(record):
    require(record.namespace == 5 and len(record.signature) == 80, "not a D16 record")
    shape = struct.unpack("<10Q", record.signature)
    require(shape[:4] == (5, 1, 0x8d41, 0x81a5), "D16 signature contract")
    width, height = shape[4:6]
    require(0 < width <= 32768 and 0 < height <= 32768 and
            shape[6:] == (1, 16, 0, 2 * width * height), "D16 shape contract")
    c = Cursor(record.data)
    require(tuple(c.integer() for _ in range(3)) == (1, record.identity, 5), "D16 blob identity")
    require(c.blob(80) == record.signature, "D16 embedded signature")
    raw = c.blob(MAX_BLOB)
    c.done()
    require(len(raw) == 2 * width * height and len(record.data) == 120 + len(raw), "D16 raw size")
    return shape, raw


def fbo_fields(record, section):
    """Return attachment offsets relative to the typed wrapper, not native chunk.

    Locked RD: header(u32 id3,u32 body length); ResourceId u64; enum u32;
    valid bool8; Attachments count u64; 10 x (bool8 + 5 x int32 + ResourceId u64);
    DrawBuffers count u64 + 8 x enum32; ReadBuffer enum32; align to 64 bytes.
    Native metadata flags are absent in this codec's Setup(). Reject others.
    """
    require(record.namespace == 4 and section >= 0x1b, "unsupported FBO/section layout")
    c = Cursor(record.data)
    require(tuple(c.integer() for _ in range(4)) == (1, record.identity, 4, 10), "FBO typed header")
    tags = tuple(c.integer() for _ in range(10))
    require(all(x in (0, 2, 5) for x in tags), "FBO namespace tags")
    native = c.blob()
    c.done()
    require(len(record.data) == 120 + len(native), "FBO wrapper layout")
    n = Cursor(native)
    require(n.integer(4) == 3, "FBO native chunk ID or flags")
    length = n.integer(4)
    require(length == 355 and len(native) == 384, "FBO native body/alignment layout")
    require((n.integer(), n.integer(4), n.integer(1), n.integer()) ==
            (record.identity, 4, 1, 10), "FBO native identity/count")
    attachments = []
    for index in range(10):
        offset = n.at
        fields = (n.integer(1),) + tuple(n.integer(4) for _ in range(5))
        identity = n.integer()
        require(fields[0] in (0, 1), "FBO bool encoding")
        require((tags[index] == 0) == (identity == 0), "FBO null tag identity")
        attachments.append({"index": index, "tag": tags[index], "identity": identity,
                            "fields": fields, "offset": 120 + offset,
                            "identity_offset": 120 + offset + 21})
    require(n.integer() == 8, "FBO draw buffer count")
    n.take(9 * 4)
    require(n.at == length + 8, "FBO parsed body span")
    # Preserve the original serializer padding; the codec validates canonical padding.
    return attachments


def d24_fields(record, section):
    """Parse the locked TextureStateInitialData and aligned raw array.

    This negative fixture deliberately requires a 2D, single-mip D24 texture;
    it is not a general texture decoder or a restriction on the actual codec.
    No raw offsets are inferred from searching for pixel/ResourceId patterns.
    """
    require(section in (0x23, 0x24) and record.namespace == 2 and
            len(record.signature) == 16 * 8, "unsupported D24 signature/section layout")
    shape = struct.unpack("<16Q", record.signature)
    require(shape[:3] == (2, 0x0de1, 0x81a6), "not a single-mip D24 2D texture")
    width, height = shape[3:5]
    require(0 < width <= 32768 and 0 < height <= 32768 and
            shape[5:8] == (1, 0, 2) and shape[10:12] == (1, 1) and
            shape[12:] == (width, height, 1, 0x81a6), "D24 physical shape contract")
    n = Cursor(record.data)
    require(n.integer(4) == 3, "D24 native chunk ID or flags")
    body_length = n.integer(4)
    require((n.integer(), n.integer(4)) == (record.identity, 2), "D24 native identity")
    require(n.integer(4) == 0x81a6 and n.integer(1) == 0, "D24 native format/view")
    require(tuple(n.integer(4) for _ in range(7)) ==
            (width, height, 1, 0, 2, 0x0de1, 1), "D24 native shape/mip count")
    # base/max level, min/max LOD, sRGB/depth mode, compare func/mode,
    # min/mag filter, seamless: each a fixed four-byte field.
    n.take(11 * 4)
    for size in (4, 3, 4):  # swizzle, wrap, border; explicit fixed-array counts
        require(n.integer() == size, "D24 fixed-array count")
        n.take(size * 4)
    n.take(4)  # lodBias float
    require(n.integer() == 0 and n.integer(4) == 0 and n.integer(4) == 0,
            "D24 unexpected texture-buffer metadata")
    n.take(4)  # maxAniso, present since section 0x23
    require(n.at == 189, "D24 fixed metadata span")
    count = n.integer()
    require(count == width * height * 4 and count <= MAX_BLOB, "D24 raw capacity")
    aligned = (n.at + 63) & ~63
    padding = aligned - n.at
    require(n.take(padding) == bytes(padding), "D24 raw alignment padding")
    offset = n.at
    raw = n.take(count)
    require(n.at == body_length + 8, "D24 native body span")
    final = (n.at + 63) & ~63
    padding = final - n.at
    require(len(record.data) == final and n.take(padding) == bytes(padding),
            "D24 native trailing padding")
    n.done()
    require(all(value == ((value >> 8) << 8 | (value >> 24))
                for (value,) in struct.iter_unpack("<I", raw)), D24_CANONICAL_ERROR)
    return shape, raw, offset


def d24_mutations(archive):
    candidates = [(i, r) for i, r in enumerate(archive.records)
                  if r.namespace == 2 and len(r.signature) >= 24 and
                  struct.unpack_from("<3Q", r.signature) == (2, 0x0de1, 0x81a6)]
    require(candidates, "fixture needs a single-mip D24 2D texture")
    index, record = candidates[0]
    _, raw, offset = d24_fields(record, archive.section)
    earlier = [(i, r.identity) for i, r in enumerate(archive.records[:index])
               if r.namespace == 2 and r.data]
    require(earlier, "D24 fixture needs an earlier nonempty writable texture")
    cases = []
    for label, word in (("first", 0), ("last", len(raw) // 4 - 1)):
        at = offset + 4 * word
        old = struct.unpack_from("<I", record.data, at)[0]
        # Preserve the high 24 bits; alter exactly one low-bit that must equal
        # the canonical replicated high byte. This is not a legal depth edit.
        changed = replace(record, data=set_word(record.data, at, old ^ 1, 4))
        records = list(archive.records)
        records[index] = changed
        cases.append(Mutation("d24-" + label + "-noncanonical-low8",
                              replace(archive, records=tuple(records)), D24_CANONICAL_ERROR,
                              {"record_index": index, "resource_id": record.identity,
                               "namespace": 2, "raw_offset": offset, "raw_word": word,
                               "original_u32": old, "mutated_u32": old ^ 1,
                               "earlier_texture_index": earlier[0][0],
                               "earlier_texture_id": earlier[0][1]}))
    return cases


def buffer_fields(record):
    require(record.namespace == 6 and len(record.signature) == 32, "buffer signature layout")
    shape = struct.unpack("<4Q", record.signature)
    require(shape[0] == 6 and shape[1] <= MAX_BLOB and shape[3] in BUFFER_TARGETS,
            "buffer signature fields")
    n = Cursor(record.data)
    require(n.integer(4) == 3, "buffer native chunk ID or flags")
    body_length = n.integer(4)
    require((n.integer(), n.integer(4)) == (record.identity, 6), "buffer native identity")
    require(n.integer(4) == shape[1] and n.integer() == shape[1], "buffer two raw capacity fields")
    padding = (-n.at) % 64
    require(n.take(padding) == bytes(padding), "buffer raw alignment padding")
    raw = n.take(shape[1])
    require(n.at == body_length + 8, "buffer native body span")
    padding = (-n.at) % 64
    require(n.take(padding) == bytes(padding), "buffer trailing padding")
    n.done()
    return shape, raw


def buffer_target_mutations(archive):
    candidates = [(i, r) for i, r in enumerate(archive.records) if r.namespace == 6 and
                  any(t.namespace == 2 and t.data for t in archive.records[:i])]
    require(candidates, "fixture needs a buffer after an earlier nonempty writable texture")
    index, record = candidates[0]
    shape, _ = buffer_fields(record)
    earlier = next((i, r.identity) for i, r in enumerate(archive.records[:index])
                   if r.namespace == 2 and r.data)
    cases = []
    for name, word, value, diagnostic in (
            ("buffer-invalid-last-target", 3, 0x0de1, BUFFER_TARGET_ERROR),
            ("buffer-queried-size-mismatch", 1, shape[1] ^ 1, STORAGE_ERROR),
            ("buffer-cached-size-mismatch", 2, shape[2] ^ 1, STORAGE_ERROR)):
        changed = replace(record, signature=set_word(record.signature, word * 8, value))
        records = list(archive.records)
        records[index] = changed
        cases.append(Mutation(name, replace(archive, records=tuple(records)), diagnostic,
                              {"record_index": index, "resource_id": record.identity,
                               "namespace": 6, "signature_word": word,
                               "original_u64": shape[word], "mutated_u64": value,
                               "earlier_texture_index": earlier[0], "earlier_texture_id": earlier[1]}))
    return cases


@dataclass(frozen=True)
class Mutation:
    name: str
    archive: Archive
    diagnostic: str
    details: dict


def mutations(archive):
    records = archive.records
    textures = [(i, r) for i, r in enumerate(records) if r.namespace == 2 and r.data]
    buffers = [(i, r) for i, r in enumerate(records) if r.namespace == 6]
    candidates = [(i, r) for i, r in enumerate(records) if r.namespace == 5 and
                  any(ti < i for ti, _ in textures)]
    require(textures and buffers and candidates,
            "fixture needs a buffer and a D16 RB after an earlier nonempty writable texture record")
    ri, rb = candidates[0]
    ti, texture = next((i, r) for i, r in textures if i < ri)
    _, raw = d16_fields(rb)
    fbos = [(i, r, a) for i, r in enumerate(records) if r.namespace == 4
            for a in fbo_fields(r, archive.section) if a["tag"] == 5 and a["identity"] == rb.identity]
    require(fbos, "fixture has no typed FBO depth reference to selected D16 RB")
    fi, fbo, attachment = fbos[0]
    require(attachment["index"] == 8 and attachment["fields"] == (0, 0, 0, 0, 0, 0),
            "original RB attachment must be single-sample, unlayered depth with zero subresource fields")
    cases = []

    def add(name, index, changed, diagnostic, **extra):
        result = list(records)
        require(changed != records[index], "unchanged mutation")
        result[index] = changed
        details = {"record_index": index, "resource_id": records[index].identity,
                   "namespace": records[index].namespace, "earlier_texture_index": ti,
                   "earlier_texture_id": texture.identity, **extra}
        cases.append(Mutation(name, replace(archive, records=tuple(result)), diagnostic, details))

    for name, offset, value in (("rb-codec", 0, 2), ("rb-embedded-id", 8, texture.identity),
                                ("rb-embedded-namespace", 16, 2)):
        add(name, ri, replace(rb, data=set_word(rb.data, offset, value)), D16_ID_ERROR)
    for name, word, value in (("format", 3, 0x81a6), ("width", 4, 0),
                               ("height", 5, 32769), ("samples", 6, 2),
                               ("depth-bits", 7, 24), ("stencil-bits", 8, 8)):
        add("rb-embedded-" + name, ri,
            replace(rb, data=set_word(rb.data, 32 + 8 * word, value)), D16_SIGNATURE_ERROR)
    # Re-encode the inner raw length too: this reaches the exact raw-capacity check,
    # rather than just a truncated outer resource blob or envelope checksum gate.
    shorter = rb.data[:112] + blob(raw[:-2])
    add("rb-short-raw-late-record", ri, replace(rb, data=shorter),
        "snapshot D16 raw payload is incomplete", later_bad_record_after_good_texture=True)
    add("rb-long-raw", ri, replace(rb, data=rb.data[:112] + blob(raw + b"\0\0")),
        "snapshot invalid blob length")
    add("rb-duplicate-top-level-id", ri, replace(rb, identity=texture.identity),
        "snapshot missing, duplicate, or foreign resource identity")
    add("rb-top-level-namespace", ri, replace(rb, namespace=2), "snapshot resource namespace mismatch")
    add("rb-top-level-signature-format", ri,
        replace(rb, signature=set_word(rb.signature, 24, 0x81a6)), STORAGE_ERROR)
    tag_offset = 32 + 8 * attachment["index"]
    id_offset = attachment["identity_offset"]
    missing_id = (1 << 64) - 1
    require(all(r.identity != missing_id for r in records), "missing-ID sentinel is in the seed inventory")
    for name, value, diagnostic in (
            ("fbo-rb-id-is-texture", texture.identity, FBO_TYPE_ERROR),
            ("fbo-rb-id-is-buffer", buffers[0][1].identity, FBO_TYPE_ERROR),
            ("fbo-rb-id-missing", missing_id,
             "snapshot references a missing ResourceId")):
        add(name, fi, replace(fbo, data=set_word(fbo.data, id_offset, value)), diagnostic,
            attachment_index=attachment["index"], expected_tag=5)
    add("fbo-illegal-namespace-tag", fi, replace(fbo, data=set_word(fbo.data, tag_offset, 6)),
        "snapshot framebuffer attachment namespace tag is unsupported")
    add("fbo-wrong-typed-count", fi, replace(fbo, data=set_word(fbo.data, 24, 9)),
        "snapshot framebuffer typed attachment count mismatch")
    for j, name in enumerate(("layered", "layer", "level", "virtual-samples", "views", "start-view")):
        offset = attachment["offset"] + (0 if j == 0 else 1 + (j - 1) * 4)
        add("fbo-rb-" + name, fi,
            replace(fbo, data=set_word(fbo.data, offset, 1, 1 if j == 0 else 4)),
            FBO_SUBRESOURCE_ERROR, attachment_index=attachment["index"])
    require(len({case.name for case in cases}) == len(cases), "duplicate mutation names")
    return cases


def reseal_manifest(original, data, state_path):
    result = copy.deepcopy(original)
    require(result.get("snapshot_api_version") == 2 and data[:8] == MAGIC, "mutation must preserve v2")
    require(result.get("state", {}).get("path") == "state.bin", "manifest state path")
    receipt = result["engine_receipt"]
    require(receipt.get("snapshot_api_version") == 2, "embedded receipt must be v2")
    identity = {"sha256": sha(data), "size_bytes": len(data)}
    result["state"].update(identity)
    receipt["snapshot_state_sha256"] = identity["sha256"]
    receipt["state_path"] = str(state_path)
    result["engine_receipt_sha256"] = sha(canonical(receipt))
    return result


def assert_rejection(code, expected, child_stderr, child_stdout, wrapper_stderr, output):
    require(code != 0, "negative archive unexpectedly succeeded")
    require("player timed out" not in wrapper_stderr, "timeout is not codec rejection")
    require("Loading and verifying complete checkpoint state" in child_stderr,
            "wrapper/engine rejected before checkpoint Load")
    lines = [line.strip() for line in child_stderr.splitlines()
             if line.startswith("DrawList replay failed:")]
    # GLReplay::LoadDrawlistSnapshot adds the lower-case prefix; the player adds
    # its own outer prefix. Check both exactly, not an arbitrary substring.
    require(lines == ["DrawList replay failed: Snapshot load refused: snapshot load refused: " + expected],
            "unexpected or non-exact codec diagnostic: " + repr(lines))
    require("Executing ordered events " not in child_stderr + child_stdout and
            "Completed replay audit" not in child_stderr + child_stdout,
            "suffix execution occurred after invalid checkpoint")
    for path in ("replay.json", "snapshot/manifest.json", "snapshot/state.bin", "color.rgba"):
        require(not (output / path).exists(), "invalid checkpoint published artifact: " + path)
    require((output / "invocation.json").is_file(), "missing real player invocation")
    for path in ("model.jsonl", "driver-command.txt", "driver-counter.txt"):
        target = output / path
        require(not target.exists() or target.stat().st_size == 0,
                "unexpected actual suffix/model evidence: " + path)


def ordering_evidence(codec_path, safety_path, depth24=False, buffer_target=False):
    code = codec_path.read_text()
    safety = safety_path.read_text()
    start = code.index("static void Load(")
    load = code[start:]
    preflight = load.index("for(const auto &record : records) SafetyPreflightResource(d, record);")
    restore = load.index("if(binding == bool(pass)) RestoreResource(d, record);")
    require(preflight < restore and "DecodeD16Blob(record); return;" in safety and
            "DecodeFramebufferBlob(d, record)" in safety, "source preflight ordering contract not found")
    if depth24:
        require("if(depth24) replay_snapshot::ValidateDepth24(data, size_t(count));" in safety and
                "texture.internalformat == eGL_DEPTH_COMPONENT24);" in safety,
                "D24 canonical raw preflight source contract not found")
    if buffer_target:
        require("BufferSignature::SameStorage(record.signature, currentSignature)" in code and
                "d.m_Buffers.at(id).curType = BufferSignature::Decode(record.signature).target;" in code,
                "buffer signature/metadata restore source contract not found")
    return {"codec_sha256": sha(code.encode()), "safety_sha256": sha(safety.encode()),
            "load_preflight_line": code[:start + preflight].count("\n") + 1,
            "load_restore_line": code[:start + restore].count("\n") + 1,
            "claim": "Source-order check only: all nested preflights precede RestoreResource loop. "
                     "Not a dynamic all-resource no-write proof or an independent whole-code review."}


def load_wrapper(path):
    spec = importlib.util.spec_from_file_location("snapshot_rejection_official_wrapper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_integration(args):
    require(math.isfinite(args.timeout) and 0 < args.timeout <= 600, "timeout must be within (0,600]")
    wrapper = Path(args.wrapper).resolve(strict=True)
    w = load_wrapper(wrapper)
    args.resume = str(w.safe_path(args.snapshot_dir))
    args.allow_model_change = False
    args.verify_color = False
    capture = w.file_identity(w.safe_path(args.capture))
    runtime = w.runtime(args)
    manifest, directory, changed = w.load_resume(args, capture, runtime)
    require(not changed and manifest["snapshot_api_version"] == 2, "need verified same-runtime v2 seed")
    data = (directory / "state.bin").read_bytes()
    archive = Archive.decode(data)
    require(archive.capture_sha == capture["sha256"] and
            archive.event == manifest["boundary"]["after_event"], "seed archive capture/boundary mismatch")
    if args.depth24_mode:
        cases, mode = d24_mutations(archive), "depth24"
    elif args.buffer_target_mode:
        cases, mode = buffer_target_mutations(archive), "buffer-target"
    else:
        cases, mode = mutations(archive), "d16-framebuffer"
    if args.case:
        require(set(args.case) <= {case.name for case in cases}, "unknown mutation case")
        cases = [case for case in cases if case.name in args.case]
    out = w.safe_path(args.outdir)
    require(not out.exists() and out.parent.is_dir(), "output must be fresh with an existing parent")
    protected_paths = [directory / "manifest.json", directory / "state.bin", wrapper, Path(__file__),
                       Path(args.codec_source), Path(args.safety_source), Path(capture["path"])]
    protected_paths += [Path(runtime[key]["path"]) for key in w.RUNTIME_FILES if runtime[key] is not None]
    protected_paths += [Path(path) for path in args.protect]
    before = [w.file_identity(path) for path in protected_paths]
    ordering = ordering_evidence(Path(args.codec_source), Path(args.safety_source),
                                 args.depth24_mode, args.buffer_target_mode)
    out.mkdir(mode=0o700)
    report = {"schema": "pvrgpu.drawlist-snapshot-rejection.v1", "status": "RUNNING",
              "original_seed": before[:2], "protected_inputs": before, "source_order": ordering,
              "positive_seed_claim": "Caller-supplied verified checkpoint; not rerun by this negative harness",
              "runtime": runtime, "mode": mode, "cases": [],
              "limitations": ["Mutated embedded receipts are resealed test inputs, not genuine success receipts.",
                              "Absence of receipts/model commands does not prove absence of raw GL resource writes.",
                              "Restore driver logs are retained and hashed, not treated as exhaustive write instrumentation."]}
    try:
        for case in cases:
            case_dir = out / case.name
            case_dir.mkdir()
            seed = case_dir / "mutated-checkpoint"
            seed.mkdir()
            payload = case.archive.encode()
            # Interior may contain the deliberate duplicate ID; only envelope/field structure is checked here.
            require(Archive.decode(payload, unique=False).encode() == payload, "mutated outer shape invalid")
            (seed / "state.bin").write_bytes(payload)
            altered = reseal_manifest(manifest, payload, seed / "state.bin")
            (seed / "manifest.json").write_bytes(canonical(altered))
            local_args = copy.copy(args)
            local_args.resume = str(seed)
            w.load_resume(local_args, capture, runtime)  # assert wrapper checks really accept this mutation
            input_before = [w.file_identity(seed / name) for name in ("manifest.json", "state.bin")]
            output = case_dir / "load"
            command = [sys.executable, str(wrapper), args.capture, "--player", args.player,
                       "--renderdoc-lib", args.renderdoc_lib, "--mesa-prefix", args.mesa_prefix,
                       "--backend", args.backend, "--gles-version", args.gles_version,
                       "--through-draw", str(args.through_draw), "--resume", str(seed),
                       "--outdir", str(output), "--timeout", str(args.timeout)]
            if args.bridge:
                command += ["--bridge", args.bridge]
            # The official wrapper owns process-group timeout/cleanup for its child.
            with (case_dir / "wrapper.stdout.log").open("xb") as stdout, \
                    (case_dir / "wrapper.stderr.log").open("xb") as stderr:
                result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
            entry = {"name": case.name, "expected_diagnostic": case.diagnostic,
                     "mutation": case.details, "argv": command, "wrapper_exit_code": result.returncode,
                     "mutated_inputs": input_before, "status": "FAIL"}
            report["cases"].append(entry)
            log = lambda name: (output / name).read_text(errors="replace") if (output / name).is_file() else ""
            assert_rejection(result.returncode, case.diagnostic, log("stderr.log"), log("stdout.log"),
                             (case_dir / "wrapper.stderr.log").read_text(errors="replace"), output)
            require([w.file_identity(seed / name) for name in ("manifest.json", "state.bin")] == input_before,
                    "mutated checkpoint changed during refusal")
            entry["evidence"] = [w.file_identity(path) for path in sorted(output.rglob("*"))
                                 if path.is_file() and not path.is_symlink() and path.stat().st_size]
            entry["status"] = "EXPECTED_CODEC_REJECTION"
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = str(error)
        raise
    finally:
        after = [w.file_identity(path) for path in protected_paths]
        report["protected_inputs_unchanged"] = before == after
        if before != after:
            report["status"] = "FAIL"
        (out / "report.json").write_bytes(canonical(report))
    require(report["status"] == "PASS", "integration audit failed")
    print(str(out / "report.json"))


def synthetic_archive():
    """Typed serializer-shaped bytes, explicitly not a real Save/Load fixture."""
    shape = b"".join(u64(x) for x in (5, 1, 0x8d41, 0x81a5, 7, 5, 1, 16, 0, 70))
    rb = Record(30, 5, shape, (), u64(1) + u64(30) + u64(5) + blob(shape) + blob(bytes(range(70))))
    native = u64(40) + struct.pack("<I", 4) + b"\x01" + u64(10)
    for index in range(10):
        native += b"\0" + bytes(20) + u64(30 if index == 8 else 10 if index == 0 else 0)
    native += u64(8) + bytes(36)
    require(len(native) == 355, "synthetic FBO layout")
    native = struct.pack("<II", 3, len(native)) + native
    native += bytes(384 - len(native))
    tags = b"".join(u64(5 if i == 8 else 2 if i == 0 else 0) for i in range(10))
    fbo = Record(40, 4, b"opaque-signature", (), u64(1) + u64(40) + u64(4) + u64(10) + tags + blob(native))
    prefix = blob(b"a" * 64) + u64(0x25) + u64(12) + u64(256)
    prefix += b"".join(blob(x) for x in (b"vendor", b"renderer", b"version", b"opaque-state"))
    prefix += u64(0) * 25
    return Archive(prefix, 0x25, 12, "a" * 64,
                   (Record(10, 2, b"texture", (), b"nonempty-writable-texture"),
                    Record(20, 6, b"buffer", (), b"nonempty-buffer"), rb, fbo))


def synthetic_d24_archive():
    """Actual fixed serializer field widths, never a live checkpoint claim."""
    source = synthetic_archive()
    width, height, identity = 7, 5, 50
    signature = b"".join(u64(x) for x in
                         (2, 0x0de1, 0x81a6, width, height, 1, 0, 2, 0, 0, 1, 1,
                          width, height, 1, 0x81a6))
    native = struct.pack("<IIQI", 3, 0, identity, 2)
    native += struct.pack("<IB7I", 0x81a6, 0, width, height, 1, 0, 2, 0x0de1, 1)
    native += bytes(11 * 4)
    for size in (4, 3, 4):
        native += u64(size) + bytes(size * 4)
    native += bytes(4 + 8 + 4 + 4) + struct.pack("<f", 1.0)
    require(len(native) == 189, "synthetic D24 metadata layout")
    codes = [0, 1, 159, 65535, 65536, 0xfffffe, 0xffffff] * 5
    raw = b"".join(struct.pack("<I", (z << 8) | (z >> 16)) for z in codes)
    native += u64(len(raw))
    native += bytes((-len(native)) % 64)
    native += raw
    native = set_word(native, 4, len(native) - 8, 4)
    native += bytes((-len(native)) % 64)
    record = Record(identity, 2, signature, (), native)
    return replace(source, section=0x24, prefix=set_word(source.prefix, 72, 0x24),
                   records=(*source.records, record))


def synthetic_buffer_archive():
    source = synthetic_archive()
    identity, size, target = 50, 32, 0x8f3f
    signature = b"".join(u64(x) for x in (6, size, size, target))
    native = struct.pack("<IIQIIQ", 3, 0, identity, 6, size, size)
    native += bytes((-len(native)) % 64) + bytes(range(size))
    native = set_word(native, 4, len(native) - 8, 4)
    native += bytes((-len(native)) % 64)
    # Keep only this real serializer-shaped buffer as a candidate; other
    # resource bytes are still explicitly synthetic, not GL success evidence.
    return replace(source, records=tuple(r for r in source.records if r.namespace != 6) +
                   (Record(identity, 6, signature, (), native),))


class SnapshotRejectionTests(unittest.TestCase):
    def test_outer_roundtrip_and_late_resource_order(self):
        source = synthetic_archive()
        self.assertEqual(Archive.decode(source.encode()), source)
        cases = mutations(source)
        self.assertEqual(len(cases), 25)
        for case in cases:
            with self.subTest(case=case.name):
                encoded = case.archive.encode()
                self.assertNotEqual(encoded, source.encode())
                self.assertEqual(encoded[:8], MAGIC)
                self.assertEqual(Archive.decode(encoded, unique=False), case.archive)
                self.assertEqual(len(source.records), len(case.archive.records))
                self.assertEqual(sum(a != b for a, b in zip(source.records, case.archive.records)), 1)
        late = next(c for c in cases if c.name == "rb-short-raw-late-record")
        self.assertLess(late.details["earlier_texture_index"], late.details["record_index"])
        self.assertEqual(late.archive.records[0], source.records[0])

    def test_checksum_corruption_refused_and_duplicate_deliberate(self):
        data = bytearray(synthetic_archive().encode())
        data[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            Archive.decode(data)
        duplicate = next(c for c in mutations(synthetic_archive()) if "duplicate" in c.name)
        with self.assertRaisesRegex(ValueError, "duplicate identity"):
            Archive.decode(duplicate.archive.encode())
        self.assertEqual(Archive.decode(duplicate.archive.encode(), unique=False), duplicate.archive)

    def test_truncation_version_and_trailing_refused(self):
        data = synthetic_archive().encode()
        for n in (0, 7, 47, 48, len(data) - 1):
            with self.subTest(n=n), self.assertRaises(ValueError):
                Archive.decode(data[:n])
        with self.assertRaisesRegex(ValueError, "magic"):
            Archive.decode(b"RDGLSN01" + data[8:])
        with self.assertRaisesRegex(ValueError, "payload size"):
            Archive.decode(data + b"\0")

    def test_exact_fbo_offsets_and_unknown_layout_refusal(self):
        source = synthetic_archive()
        fbo = source.records[3]
        fields = fbo_fields(fbo, source.section)
        self.assertEqual(fields[8]["identity"], 30)
        self.assertEqual(fields[8]["identity_offset"], 120 + 29 + 8 * 29 + 21)
        for offset, width, value in ((120, 4, 0x10003), (124, 4, 354), (24, 8, 9)):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                fbo_fields(replace(fbo, data=set_word(fbo.data, offset, value, width)), source.section)
        with self.assertRaises(ValueError):
            fbo_fields(fbo, 0x1a)

    def test_internal_field_mutations_leave_storage_inventory_exact(self):
        source = synthetic_archive()
        for case in mutations(source):
            if "embedded" in case.name or "raw" in case.name:
                with self.subTest(case=case.name):
                    self.assertEqual(case.archive.records[2].signature, source.records[2].signature)
                    self.assertEqual(case.archive.records[2].identity, source.records[2].identity)
        altered = next(c for c in mutations(source) if c.name == "fbo-rb-id-is-texture")
        attachment = fbo_fields(altered.archive.records[3], source.section)[8]
        self.assertEqual((attachment["tag"], attachment["identity"]), (5, 10))

    def test_missing_fixture_requirements_do_not_skip(self):
        source = synthetic_archive()
        for records in ((source.records[2], source.records[0], *source.records[3:]),
                        tuple(r for r in source.records if r.namespace != 6)):
            with self.assertRaisesRegex(ValueError, "fixture needs"):
                mutations(replace(source, records=records))

    def test_manifest_reseals_both_hash_layers_without_mutating_original(self):
        original = {"snapshot_api_version": 2, "state": {"path": "state.bin", "sha256": "b" * 64,
                    "size_bytes": 123}, "engine_receipt": {"snapshot_api_version": 2,
                    "state_path": "/original/state.bin", "snapshot_state_sha256": "b" * 64},
                    "engine_receipt_sha256": "c" * 64, "runtime": {"protected": "yes"}}
        before = canonical(original)
        data = mutations(synthetic_archive())[0].archive.encode()
        updated = reseal_manifest(original, data, Path("/new/state.bin"))
        self.assertEqual(canonical(original), before)
        self.assertEqual(updated["state"]["sha256"], sha(data))
        self.assertEqual(updated["state"]["size_bytes"], len(data))
        self.assertEqual(updated["engine_receipt"]["snapshot_state_sha256"], sha(data))
        self.assertEqual(updated["engine_receipt_sha256"], sha(canonical(updated["engine_receipt"])))
        self.assertEqual(updated["runtime"], original["runtime"])
        with self.assertRaises(ValueError):
            reseal_manifest(original, b"RDGLSN01" + data[8:], Path("/new/state.bin"))

    def test_rejection_contract_rejects_wrong_phase_receipt_timeout_or_suffix(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "invocation.json").write_text("{}")
            expected = D16_ID_ERROR
            stderr = ("Loading and verifying complete checkpoint state\n"
                      "DrawList replay failed: Snapshot load refused: snapshot load refused: " + expected + "\n")
            assert_rejection(1, expected, stderr, "", "player exited 1", output)
            bad = ((0, stderr, "", ""), (1, "wrapper rejected hash", "", ""),
                   (1, stderr.replace(expected, "different error"), "", ""),
                   (1, stderr + "Executing ordered events 13..20\n", "", ""),
                   (1, stderr, "Completed replay audit", ""),
                   (1, stderr, "", "player timed out"))
            for args in bad:
                with self.subTest(args=args), self.assertRaises(ValueError):
                    assert_rejection(args[0], expected, *args[1:], output)
            for name in ("replay.json", "snapshot/manifest.json", "snapshot/state.bin", "model.jsonl"):
                path = output / name
                path.parent.mkdir(exist_ok=True)
                path.write_text("unexpected")
                with self.subTest(name=name), self.assertRaises(ValueError):
                    assert_rejection(1, expected, stderr, "", "", output)
                path.unlink()

    def test_d24_exact_native_offsets_and_raw_endpoints(self):
        source = synthetic_d24_archive()
        self.assertEqual(Archive.decode(source.encode()), source)
        shape, raw, offset = d24_fields(source.records[-1], source.section)
        self.assertEqual((shape[3], shape[4], offset, len(raw)), (7, 5, 256, 140))
        self.assertEqual(struct.unpack_from("<I", raw, 0)[0], 0)
        self.assertEqual(struct.unpack_from("<I", raw, len(raw) - 4)[0], 0xffffffff)

    def test_d24_first_last_low_bits_only_and_resealed_envelope(self):
        source = synthetic_d24_archive()
        cases = d24_mutations(source)
        self.assertEqual([case.name for case in cases],
                         ["d24-first-noncanonical-low8", "d24-last-noncanonical-low8"])
        for case in cases:
            with self.subTest(case=case.name):
                self.assertEqual(Archive.decode(case.archive.encode()), case.archive)
                self.assertEqual(case.archive.records[:-1], source.records[:-1])
                self.assertEqual(case.archive.records[-1].signature, source.records[-1].signature)
                self.assertEqual(case.details["original_u32"] >> 8, case.details["mutated_u32"] >> 8)
                self.assertLess(case.details["earlier_texture_index"], case.details["record_index"])
                with self.assertRaisesRegex(ValueError, D24_CANONICAL_ERROR):
                    d24_fields(case.archive.records[-1], source.section)
        # Opt-in D24 does not alter the existing default case set.
        self.assertEqual(len(mutations(synthetic_archive())), 25)

    def test_d24_unknown_layout_padding_counts_and_truncation_refuse(self):
        source = synthetic_d24_archive()
        record = source.records[-1]
        for offset, width, value in ((0, 4, 0x10003), (4, 4, 387), (16, 4, 5),
                                     (24, 1, 1), (49, 4, 2), (97, 8, 3),
                                     (169, 8, 1), (189, 8, 136), (197, 1, 1),
                                     (447, 1, 1)):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                d24_fields(replace(record, data=set_word(record.data, offset, value, width)), source.section)
        for size in (0, 188, 255, 395, 447):
            with self.subTest(size=size), self.assertRaises(ValueError):
                d24_fields(replace(record, data=record.data[:size]), source.section)
        for section in (0x22, 0x25):
            with self.subTest(section=section), self.assertRaises(ValueError):
                d24_fields(record, section)
        with self.assertRaises(ValueError):
            d24_fields(replace(record, signature=set_word(record.signature, 88, 2)), source.section)

    def test_d24_fixture_missing_or_invalid_does_not_skip(self):
        with self.assertRaisesRegex(ValueError, "fixture needs"):
            d24_mutations(synthetic_archive())
        source = synthetic_d24_archive()
        with self.assertRaisesRegex(ValueError, "earlier"):
            d24_mutations(replace(source, records=(source.records[-1],)))
        bad = replace(source.records[-1], data=set_word(source.records[-1].data, 256, 1, 4))
        with self.assertRaisesRegex(ValueError, D24_CANONICAL_ERROR):
            d24_mutations(replace(source, records=(*source.records[:-1], bad)))

    def test_buffer_target_mutations_keep_raw_and_other_records_exact(self):
        source = synthetic_buffer_archive()
        self.assertEqual(Archive.decode(source.encode()), source)
        shape, raw = buffer_fields(source.records[-1])
        self.assertEqual(shape, (6, 32, 32, 0x8f3f))
        self.assertEqual(raw, bytes(range(32)))
        cases = buffer_target_mutations(source)
        self.assertEqual(len(cases), 3)
        for case in cases:
            with self.subTest(case=case.name):
                self.assertEqual(Archive.decode(case.archive.encode()), case.archive)
                self.assertEqual(case.archive.records[:-1], source.records[:-1])
                self.assertEqual(case.archive.records[-1].data, source.records[-1].data)
                self.assertLess(case.details["earlier_texture_index"], case.details["record_index"])
        self.assertEqual(len(mutations(synthetic_archive())), 25)

    def test_buffer_unknown_layout_capacity_and_target_refuse(self):
        source = synthetic_buffer_archive()
        record = source.records[-1]
        for offset, width, value in ((0, 4, 0x10003), (4, 4, 87), (16, 4, 2),
                                     (20, 4, 31), (24, 8, 31), (32, 1, 1), (127, 1, 1)):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                buffer_fields(replace(record, data=set_word(record.data, offset, value, width)))
        for target in (1, 0x0de1, 0xffffffff, 0x1000090d2, (1 << 64) - 1):
            with self.subTest(target=target), self.assertRaises(ValueError):
                buffer_fields(replace(record, signature=set_word(record.signature, 24, target)))
        for size in (0, 20, 63, 95, 127):
            with self.subTest(size=size), self.assertRaises(ValueError):
                buffer_fields(replace(record, data=record.data[:size]))
        with self.assertRaisesRegex(ValueError, "fixture needs"):
            buffer_target_mutations(replace(source, records=(record,)))


def main():
    if len(sys.argv) == 1 or sys.argv[1] != "run":
        unittest.main()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    for name in ("snapshot-dir", "capture", "player", "renderdoc-lib", "mesa-prefix",
                 "outdir", "codec-source", "safety-source"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--backend", choices=("pvrgpu", "llvmpipe"), required=True)
    parser.add_argument("--gles-version", choices=("3.1", "3.2"), default="3.1")
    parser.add_argument("--bridge")
    parser.add_argument("--through-draw", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--wrapper", default=str(ROOT / "script/run_drawlist_replay.py"))
    parser.add_argument("--case", action="append", help="optional exact case name; defaults to all")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--depth24-mode", action="store_true",
                       help="run two D24 canonical raw negatives; default D16/FBO cases are unchanged")
    modes.add_argument("--buffer-target-mode", action="store_true",
                       help="run three buffer target/size negatives; default D16/FBO cases are unchanged")
    parser.add_argument("--protect", action="append", default=[],
                        help="additional frozen source/lock/patch/receipt file to hash before and after")
    args = parser.parse_args()
    try:
        run_integration(args)
    except (ValueError, OSError, KeyError) as error:
        print("Snapshot rejection integration failed: " + str(error), file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
