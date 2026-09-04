#!/usr/bin/env python3
"""End-to-end checks for the functional SystemC raster pipeline."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import zlib


WIDTH = 33
HEIGHT = 35
PIXEL_COUNT = WIDTH * HEIGHT
FRAMEBUFFER_BYTES = PIXEL_COUNT * 4
SLC_LINE_BYTES = 128
# One vertex issue group plus one spatial 2x2 group per primitive. The two
# fullscreen-strip triangles share some spatial quads but cannot share a USC
# fragment group because primitive interpolation state differs.
USC_GROUP_COUNT = 1 + 323
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

STAGE_COUNTERS = (
    "vdm_cycles",
    "vertex_fetch_cycles",
    "pco_decode_cycles",
    "usc_slot_cycles",
    "usc_cluster_cycles",
    "clip_cull_cycles",
    "tiler_bin_cycles",
    "parameter_buffer_cycles",
    "tile_scheduler_cycles",
    "isp_cycles",
    "fragment_frontend_cycles",
    "texture_cycles",
    "pbe_cycles",
    "pixel_data_master_cycles",
    "slc_cycles",
    "dram_cycles",
)

UNIFIED_MEMORY_EXACT_SKIP = {
    "virtual_gpu_cycles",
    "tiler_cycles",
    "renderer_cycles",
    "pool_high_water_bytes",
    "vdm_cycles",
    "vertex_fetch_cycles",
    "parameter_buffer_cycles",
    "tile_scheduler_cycles",
    "isp_cycles",
    "fragment_frontend_cycles",
    "texture_cycles",
    "pixel_data_master_cycles",
    "pixel_data_master_transactions",
    "pixel_data_master_bytes",
    "tcu_line_accesses",
    "tcu_read_accesses",
    "tcu_hits",
    "tcu_misses",
    "tcu_evictions",
    "tcu_writebacks",
    "tcu_bypassed",
    "tcu_cycles",
    "slc_line_accesses",
    "slc_read_accesses",
    "slc_write_accesses",
    "slc_hits",
    "slc_misses",
    "slc_evictions",
    "slc_writebacks",
    "slc_bypassed",
    "slc_cycles",
    "dram_read_transactions",
    "dram_write_transactions",
    "dram_read_bytes",
    "dram_write_bytes",
    "dram_cycles",
    "memory_direct_read_bytes",
    "memory_direct_write_bytes",
    "framebuffer_dram_readback_bytes",
}


def verify_memory_path(
    hello: dict[str, object],
    message: dict[str, object],
    *,
    cache_bypass: bool,
    warm_slc: bool = False,
    framebuffer_bytes: int = FRAMEBUFFER_BYTES,
) -> None:
    expected_mode = "bypass" if cache_bypass else "cache"
    assert hello.get("cache_bypass") is cache_bypass
    assert hello.get("memory_mode") == expected_mode
    assert hello.get("cache_simulated") is (expected_mode == "cache")
    assert hello.get("framebuffer_source") == "dram-readback"
    assert hello.get("dram_fixed_latency_cycles") == 1
    values = message.get("counters")
    assert isinstance(values, dict)
    common = {
        "pixel_data_master_transactions": 1,
        "pixel_data_master_bytes": framebuffer_bytes,
        "pixel_data_master_cycles": 1,
        "framebuffer_dram_readback_bytes": framebuffer_bytes,
    }
    for field, expected in common.items():
        assert values.get(field) == expected, (
            f"memory path: {field}={values.get(field)!r}, expected {expected}"
        )

    for field in (
        "tcu_line_accesses",
        "tcu_read_accesses",
        "tcu_hits",
        "tcu_misses",
        "tcu_evictions",
        "tcu_writebacks",
        "tcu_bypassed",
        "tcu_cycles",
    ):
        assert values.get(field) == 0, (
            f"unified memory path: {field}={values.get(field)!r}, expected 0"
        )

    assert values.get("memory_direct_read_bytes") == 0
    assert values.get("memory_direct_write_bytes") == 0
    assert values["dram_read_transactions"] > 0
    assert values["dram_write_transactions"] > 0
    assert values["dram_read_bytes"] >= framebuffer_bytes
    assert values["dram_write_bytes"] >= framebuffer_bytes
    assert values["dram_cycles"] == (
        values["dram_read_transactions"] + values["dram_write_transactions"]
    )

    if cache_bypass:
        for field in (
            "slc_line_accesses",
            "slc_read_accesses",
            "slc_write_accesses",
            "slc_hits",
            "slc_misses",
            "slc_evictions",
            "slc_writebacks",
            "slc_cycles",
        ):
            assert values.get(field) == 0, (
                f"bypass memory path: {field}={values.get(field)!r}, expected 0"
            )
        assert values["slc_bypassed"] > 0
        return

    assert values["slc_bypassed"] == 0
    assert values["slc_line_accesses"] > 0
    assert values["slc_line_accesses"] == (
        values["slc_read_accesses"] + values["slc_write_accesses"]
    )
    assert values["slc_hits"] + values["slc_misses"] == values[
        "slc_line_accesses"
    ]
    assert values["slc_cycles"] == values["slc_line_accesses"]
    assert values["slc_writebacks"] > 0
    assert values["dram_write_transactions"] == values["slc_writebacks"]
    assert values["dram_write_bytes"] == values["slc_writebacks"] * SLC_LINE_BYTES


def invoke_case(
    executable: Path,
    case_name: str,
    output_dir: Path,
    *,
    frames: int,
    memory_mode: str | None = None,
    cache_bypass: str | None = None,
    width: int = WIDTH,
    height: int = HEIGHT,
    timeout: int = 15,
) -> subprocess.CompletedProcess[str]:
    arguments = [
        str(executable),
        "--frames",
        str(frames),
        "--width",
        str(width),
        "--height",
        str(height),
        "--case",
        case_name,
        "--outdir",
        str(output_dir),
    ]
    if memory_mode is not None:
        arguments.extend(("--memory-mode", memory_mode))
    if cache_bypass is not None:
        arguments.extend(("--cache-bypass", cache_bypass))
    return subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def process_details(completed: subprocess.CompletedProcess[str]) -> str:
    return (
        f"exit={completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def invoke_driver_command(
    executable: Path,
    command_path: Path,
    output_dir: Path,
    *,
    cache_bypass: str = "on",
    timeout: int = 15,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(executable),
            "--driver-command",
            str(command_path),
            "--outdir",
            str(output_dir),
            "--cache-bypass",
            cache_bypass,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def json_messages(completed: subprocess.CompletedProcess[str]) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.lstrip().startswith("{")
    ]


def paeth_predictor(left: int, above: int, upper_left: int) -> int:
    prediction = left + above - upper_left
    distance_left = abs(prediction - left)
    distance_above = abs(prediction - above)
    distance_upper_left = abs(prediction - upper_left)
    if distance_left <= distance_above and distance_left <= distance_upper_left:
        return left
    if distance_above <= distance_upper_left:
        return above
    return upper_left


def decode_rgba8_png(path: Path) -> tuple[int, int, bytes]:
    data = path.read_bytes()
    assert data.startswith(PNG_SIGNATURE), f"{path}: invalid PNG signature"

    offset = len(PNG_SIGNATURE)
    ihdr: tuple[int, int, int, int, int, int, int] | None = None
    idat_parts: list[bytes] = []
    saw_iend = False
    chunk_index = 0

    while offset < len(data):
        assert offset + 12 <= len(data), f"{path}: truncated PNG chunk header"
        length = struct.unpack_from(">I", data, offset)[0]
        chunk_type = data[offset + 4 : offset + 8]
        payload_start = offset + 8
        payload_end = payload_start + length
        crc_end = payload_end + 4
        assert crc_end <= len(data), f"{path}: truncated {chunk_type!r} chunk"

        payload = data[payload_start:payload_end]
        expected_crc = struct.unpack_from(">I", data, payload_end)[0]
        actual_crc = zlib.crc32(chunk_type)
        actual_crc = zlib.crc32(payload, actual_crc) & 0xFFFFFFFF
        assert actual_crc == expected_crc, f"{path}: bad {chunk_type!r} CRC"

        if chunk_type == b"IHDR":
            assert chunk_index == 0, f"{path}: IHDR is not the first chunk"
            assert ihdr is None, f"{path}: duplicate IHDR"
            assert length == 13, f"{path}: invalid IHDR length"
            ihdr = struct.unpack(">IIBBBBB", payload)
        elif chunk_type == b"IDAT":
            assert ihdr is not None, f"{path}: IDAT precedes IHDR"
            idat_parts.append(payload)
        elif chunk_type == b"IEND":
            assert length == 0, f"{path}: invalid IEND length"
            saw_iend = True
            offset = crc_end
            break

        offset = crc_end
        chunk_index += 1

    assert ihdr is not None, f"{path}: missing IHDR"
    assert idat_parts, f"{path}: missing IDAT"
    assert saw_iend, f"{path}: missing IEND"
    assert offset == len(data), f"{path}: trailing bytes after IEND"

    width, height, bit_depth, color_type, compression, filtering, interlace = ihdr
    assert bit_depth == 8, f"{path}: expected 8-bit channels, got {bit_depth}"
    assert color_type == 6, f"{path}: expected RGBA color type, got {color_type}"
    assert compression == 0, f"{path}: unsupported compression method"
    assert filtering == 0, f"{path}: unsupported PNG filtering method"
    assert interlace == 0, f"{path}: interlaced PNG is not supported"

    decompressor = zlib.decompressobj()
    filtered = decompressor.decompress(b"".join(idat_parts))
    filtered += decompressor.flush()
    assert decompressor.eof, f"{path}: incomplete zlib stream"
    assert not decompressor.unused_data, f"{path}: trailing zlib stream data"
    assert not decompressor.unconsumed_tail, f"{path}: unconsumed zlib input"

    bytes_per_pixel = 4
    row_bytes = width * bytes_per_pixel
    expected_filtered_bytes = height * (row_bytes + 1)
    assert len(filtered) == expected_filtered_bytes, (
        f"{path}: expected {expected_filtered_bytes} filtered bytes, "
        f"got {len(filtered)}"
    )

    pixels = bytearray()
    previous_row = bytearray(row_bytes)
    cursor = 0
    for row_index in range(height):
        filter_type = filtered[cursor]
        cursor += 1
        encoded = filtered[cursor : cursor + row_bytes]
        cursor += row_bytes
        reconstructed = bytearray(row_bytes)

        for byte_index, encoded_byte in enumerate(encoded):
            left = (
                reconstructed[byte_index - bytes_per_pixel]
                if byte_index >= bytes_per_pixel
                else 0
            )
            above = previous_row[byte_index]
            upper_left = (
                previous_row[byte_index - bytes_per_pixel]
                if byte_index >= bytes_per_pixel
                else 0
            )

            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = above
            elif filter_type == 3:
                predictor = (left + above) // 2
            elif filter_type == 4:
                predictor = paeth_predictor(left, above, upper_left)
            else:
                raise AssertionError(
                    f"{path}: row {row_index} has invalid filter {filter_type}"
                )
            reconstructed[byte_index] = (encoded_byte + predictor) & 0xFF

        pixels.extend(reconstructed)
        previous_row = reconstructed

    return width, height, bytes(pixels)


def verify_driver_clear_command(executable: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command_path = output_dir / "driver-clear-green.txt"
    command_path.write_text(
        "\n".join(
            (
                "schema=pvrgpu.driver-command.v1",
                "producer=pvrgpu-gallium-driver",
                "command=clear_color",
                "case=phase1.clear.green",
                "frame=1",
                f"width={WIDTH}",
                f"height={HEIGHT}",
                "format=PIPE_FORMAT_R8G8B8A8_UNORM",
                "clear_color_bits=0,1065353216,0,1065353216",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    completed = invoke_driver_command(executable, command_path, output_dir)
    assert completed.returncode == 0, (
        "driver clear command run failed:\n" + process_details(completed)
    )
    messages = json_messages(completed)
    hello = [message for message in messages if message.get("type") == "hello"]
    counters = [message for message in messages if message.get("type") == "counter"]
    done = [message for message in messages if message.get("type") == "done"]
    assert len(hello) == 1, "driver clear: missing/duplicate hello"
    assert hello[0].get("mode") == "pvrgpu-driver-clear-color-phase1"
    assert hello[0].get("command_source") == "pvrgpu-gallium-driver-command"
    assert hello[0].get("driver_command_ingest") is True
    assert hello[0].get("driver_command") == "clear_color"
    assert len(counters) == 1, "driver clear: missing/duplicate counter"
    assert counters[0].get("command_source") == "pvrgpu-gallium-driver-command"
    assert counters[0].get("driver_command_ingest") is True
    assert counters[0].get("functional_scope") == "driver_clear_color-pco-iss-v1"
    assert done and done[-1].get("pool_leaks") == 0
    values = counters[0].get("counters")
    assert isinstance(values, dict)
    for field in (
        "ia_vertices",
        "ia_primitives",
        "vs_invocations",
        "gs_invocations",
        "gs_primitives",
        "c_invocations",
        "c_primitives",
        "ps_invocations",
        "hs_invocations",
        "ds_invocations",
        "cs_invocations",
        "ts_invocations",
        "ms_invocations",
        "ms_primitives",
        "drawlists",
        "setup_triangles",
        "texel_fetches",
    ):
        assert values.get(field) == 0, (
            f"driver clear: {field}={values.get(field)!r}, expected 0"
        )
    for field in (
        "vs_alu_instructions",
        "vs_tex_instructions",
        "vs_memory_instructions",
        "fs_alu_instructions",
        "fs_tex_instructions",
        "fs_memory_instructions",
    ):
        assert values.get(field) == 0, (
            f"driver clear API view: {field}={values.get(field)!r}, expected 0"
        )
    assert values.get("depth_rejected_fragments") == PIXEL_COUNT
    assert values.get("pbe_pixels_written") == PIXEL_COUNT
    assert counters[0].get("drawlist_stats") == []
    verify_memory_path(hello[0], counters[0], cache_bypass=True)

    artifact = output_dir / "driver_clear_color_sample_000001.png"
    png_width, png_height, pixels = decode_rgba8_png(artifact)
    assert (png_width, png_height) == (WIDTH, HEIGHT)
    assert pixels == b"\x00\xFF\x00\xFF" * PIXEL_COUNT, (
        "driver clear: framebuffer is not solid green"
    )


def verify_driver_triangle_command(executable: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command_path = output_dir / "driver-triangle-red.txt"
    command_path.write_text(
        "\n".join(
            (
                "schema=pvrgpu.driver-command.v1",
                "producer=pvrgpu-gallium-driver",
                "command=draw_triangle",
                "case=phase2.draw_triangle.gallium",
                "frame=1",
                f"width={WIDTH}",
                f"height={HEIGHT}",
                "format=PIPE_FORMAT_R8G8B8A8_UNORM",
                "clear_color_bits=0,0,0,1065353216",
                "vertex0_bits=3212836864,3212836864",
                "vertex1_bits=1065353216,3212836864",
                "vertex2_bits=0,1065353216",
                "fragment_color_bits=1065353216,0,0,1065353216",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    completed = invoke_driver_command(executable, command_path, output_dir)
    assert completed.returncode == 0, (
        "driver triangle command run failed:\n" + process_details(completed)
    )
    messages = json_messages(completed)
    hello = [message for message in messages if message.get("type") == "hello"]
    counters = [message for message in messages if message.get("type") == "counter"]
    done = [message for message in messages if message.get("type") == "done"]
    assert len(hello) == 1, "driver triangle: missing/duplicate hello"
    assert hello[0].get("mode") == "pvrgpu-driver-draw-triangle-phase2"
    assert hello[0].get("command_source") == "pvrgpu-gallium-driver-command"
    assert hello[0].get("driver_command_ingest") is True
    assert hello[0].get("driver_command") == "draw_triangle"
    assert hello[0].get("pco_subset") == "mbyp-uvsw-driver-triangle"
    assert len(counters) == 1, "driver triangle: missing/duplicate counter"
    assert counters[0].get("command_source") == "pvrgpu-gallium-driver-command"
    assert counters[0].get("driver_command_ingest") is True
    assert (
        counters[0].get("functional_scope")
        == "driver_triangle_solid-pco-iss-v1"
    )
    assert done and done[-1].get("pool_leaks") == 0
    values = counters[0].get("counters")
    assert isinstance(values, dict)
    assert values.get("ia_vertices") == 3
    assert values.get("ia_primitives") == 1
    assert values.get("vs_invocations") == 3
    assert values.get("c_primitives") == 1
    assert values.get("setup_triangles") == 1
    assert values.get("ps_invocations", 0) > 0
    assert values.get("pbe_pixels_written", 0) > 0
    verify_memory_path(hello[0], counters[0], cache_bypass=True)

    artifact = output_dir / "driver_triangle_solid_sample_000001.png"
    png_width, png_height, pixels = decode_rgba8_png(artifact)
    assert (png_width, png_height) == (WIDTH, HEIGHT)
    red_pixels = sum(
        1
        for offset in range(0, len(pixels), 4)
        if pixels[offset : offset + 4] == b"\xFF\x00\x00\xFF"
    )
    black_pixels = sum(
        1
        for offset in range(0, len(pixels), 4)
        if pixels[offset : offset + 4] == b"\x00\x00\x00\xFF"
    )
    assert red_pixels > 0, "driver triangle: framebuffer has no red pixels"
    assert black_pixels > 0, "driver triangle: framebuffer has no clear pixels"


def verify_driver_indexed_quad_framebuffer_size(
    executable: Path, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command_path = output_dir / "driver-indexed-quad.txt"
    framebuffer_width = 128
    framebuffer_height = 96
    viewport_width = 64
    viewport_height = 64
    draw_count = 3
    command_path.write_text(
        "\n".join(
            (
                "schema=pvrgpu.driver-command.v1",
                "producer=pvrgpu-gallium-driver",
                "command=draw_indexed_quad",
                "case=phase7.draw_indexed_quad.gallium",
                "frame=1",
                f"framebuffer_width={framebuffer_width}",
                f"framebuffer_height={framebuffer_height}",
                f"width={viewport_width}",
                f"height={viewport_height}",
                "format=PIPE_FORMAT_R8G8B8A8_UNORM",
                "clear_color_bits=0,0,0,1065353216",
                f"draw_count={draw_count}",
                "index_count=6",
                "unique_vertices=4",
                "primitive_count=2",
                "clip_primitives=2",
                "setup_triangles=2",
                "semantic_texel_fetches=12345",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    completed = invoke_driver_command(executable, command_path, output_dir)
    assert completed.returncode == 0, (
        "driver indexed quad command run failed:\n" + process_details(completed)
    )
    messages = json_messages(completed)
    hello = [message for message in messages if message.get("type") == "hello"]
    counters = [message for message in messages if message.get("type") == "counter"]
    done = [message for message in messages if message.get("type") == "done"]
    assert len(hello) == 1, "driver indexed quad: missing/duplicate hello"
    assert hello[0].get("mode") == "pvrgpu-driver-draw-indexed-quad-phase7"
    assert hello[0].get("driver_command_ingest") is True
    assert hello[0].get("driver_command") == "draw_indexed_quad"
    assert hello[0].get("driver_command_width") == viewport_width
    assert hello[0].get("driver_command_height") == viewport_height
    assert hello[0].get("driver_command_framebuffer_width") == framebuffer_width
    assert hello[0].get("driver_command_framebuffer_height") == framebuffer_height
    assert len(counters) == 1, "driver indexed quad: missing/duplicate counter"
    assert done and done[-1].get("pool_leaks") == 0
    values = counters[0].get("counters")
    assert isinstance(values, dict)
    assert values.get("ia_vertices") == 6 * draw_count
    assert values.get("ia_primitives") == 2 * draw_count
    assert values.get("vs_invocations") == 4 * draw_count
    assert values.get("c_invocations") == 2 * draw_count
    assert values.get("c_primitives") == 2 * draw_count
    assert values.get("ps_invocations") == (
        viewport_width * viewport_height * draw_count
    )
    assert values.get("drawlists") == draw_count
    assert values.get("setup_triangles") == 2 * draw_count
    # semantic_texel_fetches=12345 is only the driver's estimate.  The texture
    # unit did not sample anything for this quad, so the reported figure is the
    # measurement, not the estimate.
    assert values.get("texel_fetches") == 0
    assert values.get("fs_alu_instructions") == (
        viewport_width * viewport_height * 4 * draw_count
    )
    verify_memory_path(
        hello[0],
        counters[0],
        cache_bypass=True,
        framebuffer_bytes=framebuffer_width * framebuffer_height * 4,
    )

    drawlists = counters[0].get("drawlist_stats")
    assert isinstance(drawlists, list) and len(drawlists) == draw_count
    for drawlist in drawlists:
        fs = drawlist.get("fs")
        assert isinstance(fs, dict)
        assert fs.get("invocations") == viewport_width * viewport_height

    artifact = output_dir / "driver_indexed_quad_sample_000001.png"
    png_width, png_height, pixels = decode_rgba8_png(artifact)
    assert (png_width, png_height) == (framebuffer_width, framebuffer_height)
    assert len(pixels) == framebuffer_width * framebuffer_height * 4
    assert set(
        tuple(pixels[offset : offset + 4]) for offset in range(0, len(pixels), 4)
    ) == {(0, 0, 0, 255)}, "driver indexed quad framebuffer is not opaque black"


def verify_driver_primitive_sequence_is_rejected(
    executable: Path, output_dir: Path
) -> None:
    """A retired draw_primitive_sequence capsule must not be honoured.

    The command existed so a case-name profile could state API counters the
    pipeline never produced.  It is gone: the model must refuse the capsule
    rather than ingest the counters it carries.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    command_path = output_dir / "driver-primitive-sequence.txt"
    command_path.write_text(
        "\n".join(
            (
                "schema=pvrgpu.driver-command.v1",
                "producer=pvrgpu-gallium-driver",
                "command=draw_primitive_sequence",
                "case=dEQP-GLES3.functional.rasterization.primitives.line_loop",
                "frame=1",
                "width=512",
                "height=512",
                "format=PIPE_FORMAT_R8G8B8A8_UNORM",
                "clear_color_bits=0,0,0,1065353216",
                "draw_count=3",
                "ia_vertices=12",
                "ia_primitives=12",
                "vs_invocations=12",
                "clip_invocations=12",
                "clip_primitives=12",
                "setup_triangles=0",
                "ps_invocations=1052",
                "semantic_texel_fetches=0",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    completed = invoke_driver_command(executable, command_path, output_dir)
    assert completed.returncode != 0, (
        "a retired draw_primitive_sequence capsule was accepted:\n"
        + process_details(completed)
    )
    details = process_details(completed)
    assert "unsupported driver command" in details, details


def triangle_setup_half_culled_golden_pixels() -> bytes:
    """Rebuild the GLBench seeded winding image independently of the model."""

    # Apple/FreeBSD Libc's rand() uses this Park-Miller transition.  GLBench
    # calls srand(0), consumes one value per cell in j+=4 -> i -> j2 order,
    # and treats values below RAND_MAX/2 as front-facing.  PNG row zero is the
    # vertically flipped OpenGL framebuffer row y=63.
    state = 0
    front = [[False] * 128 for _ in range(128)]
    for swath_y in range(0, 128, 4):
        for x in range(128):
            for row_offset in range(4):
                if state == 0:
                    state = 123459876
                high, low = divmod(state, 127773)
                state = 16807 * low - 2836 * high
                if state < 0:
                    state += 0x7FFFFFFF
                front[swath_y + row_offset][x] = (
                    state < 0x7FFFFFFF // 2
                )

    pixels = bytearray()
    cyan = bytes((0, 128, 128, 255))
    green = bytes((0, 255, 0, 255))
    for png_y in range(64):
        mesh_y = 32 + (63 - png_y)
        for png_x in range(64):
            pixels.extend(cyan if front[mesh_y][32 + png_x] else green)

    result = bytes(pixels)
    assert hashlib.sha256(result).hexdigest() == (
        "0ca6e9f6474b012e9cd63b16fd5769556d6593589db74ac5863acf4dd3f5c03a"
    )
    return result


def varyings_shader_one_golden_pixels() -> bytes:
    """Rebuild GLBench v1=c varying output independently of the model."""

    pixels = bytearray()
    for png_y in range(64):
        green = max(0.0, (63 - 2 * png_y) / 64.0)
        for x in range(64):
            red = max(0.0, (2 * x - 63) / 64.0)
            pixels.extend(
                (
                    int(red * 255.0 + 0.5),
                    int(green * 255.0 + 0.5),
                    0,
                    255,
                )
            )
    result = bytes(pixels)
    assert hashlib.sha256(result).hexdigest() == (
        "5424d51eef6ebbc37e1af5b5978cf12af04ea2ae9cee77412bb45fccce787342"
    )
    return result


def varyings_shader_two_golden_pixels() -> bytes:
    """Independently evaluate GLBench v1=v2=c/2, color=v1+v2."""

    pixels = bytearray()
    for png_y in range(64):
        clip_y = (63 - 2 * png_y) / 64.0
        v1_y = clip_y * 0.5
        v2_y = clip_y * 0.5
        green = max(0.0, v1_y + v2_y)
        for x in range(64):
            clip_x = (2 * x - 63) / 64.0
            v1_x = clip_x * 0.5
            v2_x = clip_x * 0.5
            red = max(0.0, v1_x + v2_x)
            pixels.extend(
                (
                    int(red * 255.0 + 0.5),
                    int(green * 255.0 + 0.5),
                    0,
                    255,
                )
            )
    result = bytes(pixels)
    assert hashlib.sha256(result).hexdigest() == (
        "5424d51eef6ebbc37e1af5b5978cf12af04ea2ae9cee77412bb45fccce787342"
    )
    return result


def varyings_shader_four_golden_pixels() -> bytes:
    """Independently evaluate GLBench v1..v4=c/4 and left-assoc sum."""

    pixels = bytearray()
    for png_y in range(64):
        clip_y = (63 - 2 * png_y) / 64.0
        quarter_y = clip_y * 0.25
        # Preserve the GLSL source's left-associative addition tree instead of
        # reducing directly to c. This remains independent of PCO execution.
        green = max(
            0.0,
            ((quarter_y + quarter_y) + quarter_y) + quarter_y,
        )
        for x in range(64):
            clip_x = (2 * x - 63) / 64.0
            quarter_x = clip_x * 0.25
            red = max(
                0.0,
                ((quarter_x + quarter_x) + quarter_x) + quarter_x,
            )
            pixels.extend(
                (
                    int(red * 255.0 + 0.5),
                    int(green * 255.0 + 0.5),
                    0,
                    255,
                )
            )
    result = bytes(pixels)
    assert hashlib.sha256(result).hexdigest() == (
        "5424d51eef6ebbc37e1af5b5978cf12af04ea2ae9cee77412bb45fccce787342"
    )
    return result


def varyings_shader_eight_golden_pixels() -> bytes:
    """Independently evaluate GLBench v1..v8=c/8 and left-assoc sum."""

    pixels = bytearray()
    for png_y in range(64):
        clip_y = (63 - 2 * png_y) / 64.0
        eighth_y = clip_y * 0.125
        green_sum = eighth_y
        for _ in range(7):
            green_sum += eighth_y
        green = max(0.0, green_sum)
        for x in range(64):
            clip_x = (2 * x - 63) / 64.0
            eighth_x = clip_x * 0.125
            red_sum = eighth_x
            for _ in range(7):
                red_sum += eighth_x
            red = max(0.0, red_sum)
            pixels.extend(
                (
                    int(red * 255.0 + 0.5),
                    int(green * 255.0 + 0.5),
                    0,
                    255,
                )
            )
    result = bytes(pixels)
    assert hashlib.sha256(result).hexdigest() == (
        "5424d51eef6ebbc37e1af5b5978cf12af04ea2ae9cee77412bb45fccce787342"
    )
    return result


def fill_tex_nearest_golden_pixels() -> bytes:
    """Rebuild GLBench's 512x512 level-0 nearest result independently."""

    texture_size = 512
    surface_size = 64
    pixels = bytearray()
    for png_y in range(surface_size):
        # PNG row zero is the vertically flipped OpenGL framebuffer row 63.
        framebuffer_y = surface_size - 1 - png_y
        texel_y = ((2 * framebuffer_y + 1) * texture_size) // (
            2 * surface_size
        )
        for png_x in range(surface_size):
            texel_x = ((2 * png_x + 1) * texture_size) // (2 * surface_size)
            # SetupTexture(9) level zero writes R=0 and G=B=(row XOR column),
            # with conversion to GL_UNSIGNED_BYTE retaining the low 8 bits.
            fractal = (texel_y ^ texel_x) & 0xFF
            pixels.extend((0, fractal, fractal, 255))

    result = bytes(pixels)
    assert len(result) == surface_size * surface_size * 4
    colors = {
        result[offset : offset + 4] for offset in range(0, len(result), 4)
    }
    assert len(colors) == 32
    assert hashlib.sha256(result).hexdigest() == (
        "55864a1b7302851c70ea27211a275e512b42313d5177b72bf00e1c65b95737e5"
    )
    return result


def fill_tex_bilinear_golden_pixels() -> bytes:
    """Evaluate the GLES normalized-repeat, level-0 linear sample exactly."""

    texture_size = 512
    surface_size = 64

    def level_zero(texel_x: int, texel_y: int) -> int:
        # SetupTexture(9) stores R=0 and G=B=(row XOR column), retaining the
        # low eight bits when converted to GL_UNSIGNED_BYTE.
        return (texel_y ^ texel_x) & 0xFF

    pixels = bytearray()
    for png_y in range(surface_size):
        # OpenGL framebuffer row 63 is encoded as PNG row zero.
        framebuffer_y = surface_size - 1 - png_y
        normalized_v = (framebuffer_y + 0.5) / surface_size
        texture_v = normalized_v * texture_size - 0.5
        texel_y0_unwrapped = math.floor(texture_v)
        texel_y0 = texel_y0_unwrapped % texture_size
        texel_y1 = (texel_y0 + 1) % texture_size
        weight_y = texture_v - texel_y0_unwrapped

        for png_x in range(surface_size):
            normalized_u = (png_x + 0.5) / surface_size
            texture_u = normalized_u * texture_size - 0.5
            texel_x0_unwrapped = math.floor(texture_u)
            texel_x0 = texel_x0_unwrapped % texture_size
            texel_x1 = (texel_x0 + 1) % texture_size
            weight_x = texture_u - texel_x0_unwrapped

            tap00 = level_zero(texel_x0, texel_y0)
            tap10 = level_zero(texel_x1, texel_y0)
            tap01 = level_zero(texel_x0, texel_y1)
            tap11 = level_zero(texel_x1, texel_y1)
            filtered = (
                tap00 * (1.0 - weight_x) * (1.0 - weight_y)
                + tap10 * weight_x * (1.0 - weight_y)
                + tap01 * (1.0 - weight_x) * weight_y
                + tap11 * weight_x * weight_y
            )
            channel = int(filtered + 0.5)
            pixels.extend((0, channel, channel, 255))

    result = bytes(pixels)
    assert len(result) == surface_size * surface_size * 4
    assert hashlib.sha256(result).hexdigest() == (
        "b0ab230552a02a15d19d5d646f8f8d6c425d2ef4b74503226bfff8ce535c509d"
    )
    return result


def verify_fill_solid(executable: Path, output_dir: Path) -> None:
    completed = invoke_case(executable, "fill_solid", output_dir, frames=2)
    assert completed.returncode == 0, (
        "fill_solid model run failed:\n" + process_details(completed)
    )

    messages = json_messages(completed)
    hello = [message for message in messages if message.get("type") == "hello"]
    counters = [
        message for message in messages if message.get("type") == "counter"
    ]
    done = [message for message in messages if message.get("type") == "done"]

    assert len(hello) == 1, "fill_solid: missing/duplicate hello"
    assert hello[0].get("mode") == "systemc-functional-fill-solid"
    assert hello[0].get("functional_scope") == "fill_solid-pco-iss-v1"
    assert hello[0].get("shader_binary") == "mesa-pco-public-encoding"
    assert hello[0].get("pco_subset") == "mbyp-uvsw-fill-solid"
    assert hello[0].get("reference_uarch") == "pvrgpu-ref-v1"
    assert hello[0].get("uarch_provenance") == "assumed"
    assert hello[0].get("tile_width") == 32
    assert hello[0].get("tile_height") == 32
    assert hello[0].get("cache_bypass") is False
    assert hello[0].get("cache_policy") == (
        "set-associative-write-back-write-allocate-true-lru"
    )
    assert hello[0].get("mcu_cache") == {
        "capacity_bytes": 24 * 1024,
        "line_bytes": 64,
        "ways": 4,
        "banks": 4,
    }
    assert hello[0].get("tcu_cache") == {
        "capacity_bytes": 24 * 1024,
        "line_bytes": 64,
        "ways": 4,
        "banks": 4,
    }
    assert hello[0].get("slc_cache") == {
        "capacity_bytes": 2 * 1024 * 1024,
        "line_bytes": 128,
        "ways": 8,
        "banks": 8,
    }
    assert hello[0].get("usc_l2_cache") == {
        "capacity_bytes": 8 * 1024,
        "line_bytes": 64,
        "ways": 4,
        "banks": 1,
    }
    assert [message.get("frame") for message in counters] == [1, 2]
    assert len(done) == 1, "fill_solid: missing/duplicate done"
    assert done[0].get("pool_leaks") == 0, "fill_solid: MemoryPool leak"

    expected_counters = {
        "ia_vertices": 4,
        "ia_primitives": 2,
        "vs_invocations": 4,
        "c_invocations": 2,
        "c_primitives": 2,
        "ps_invocations": PIXEL_COUNT,
        "texel_fetches": 0,
        "texture_requests": 0,
        "tiles_binned": 4,
        "tiles_scheduled": 4,
        "covered_pixels": PIXEL_COUNT,
        "fragment_candidates": PIXEL_COUNT,
        "hsr_rejected_fragments": 0,
        "depth_tested_fragments": 0,
        "depth_rejected_fragments": 0,
        "depth_written_fragments": 0,
        "pco_instructions": 6,
        "vs_alu_instructions": 0,
        "vs_tex_instructions": 0,
        "vs_memory_instructions": 16,
        "fs_alu_instructions": PIXEL_COUNT * 4,
        "fs_tex_instructions": 0,
        "fs_memory_instructions": 0,
        "usc_groups": USC_GROUP_COUNT,
        "pbe_color_reads": 0,
        "pbe_blended_fragments": 0,
        "pbe_fragment_writes": PIXEL_COUNT,
        "pbe_pixels_written": PIXEL_COUNT,
    }
    for index, message in enumerate(counters):
        assert message.get("provenance") == "modeled", (
            f"fill_solid frame {message.get('frame')}: wrong provenance"
        )
        values = message.get("counters")
        assert isinstance(values, dict), "fill_solid: counters is not an object"
        for field, expected in expected_counters.items():
            if field in UNIFIED_MEMORY_EXACT_SKIP:
                continue
            assert values.get(field) == expected, (
                f"fill_solid frame {message.get('frame')}: {field}="
                f"{values.get(field)!r}, expected {expected}"
            )
        assert values["fragment_candidates"] == (
            values["ps_invocations"] + values["hsr_rejected_fragments"]
        ), "fill_solid: ISP/HSR candidate conservation failed"
        assert message.get("drawlist_stats") == [
            {
                "drawlist": 0,
                "draw_id": 0,
                "vs": {
                    "invocations": 4,
                    "program": {
                        "groups": 2,
                        "instructions": 2,
                        "alu": 0,
                        "tex": 0,
                        "memory": 2,
                    },
                    "executed": {"alu": 0, "tex": 0, "memory": 16},
                },
                "fs": {
                    "invocations": PIXEL_COUNT,
                    "program": {
                        "groups": 4,
                        "instructions": 4,
                        "alu": 4,
                        "tex": 0,
                        "memory": 0,
                    },
                    "executed": {
                        "alu": PIXEL_COUNT * 4,
                        "tex": 0,
                        "memory": 0,
                    },
                },
            }
        ], "fill_solid: per-DrawList PCO statistics mismatch"
        for field in STAGE_COUNTERS:
            value = values.get(field)
            assert isinstance(value, (int, float)) and not isinstance(value, bool)
            assert value > 0, (
                f"fill_solid frame {message.get('frame')}: {field} not active"
            )
        assert values["virtual_gpu_cycles"] == (
            values["tiler_cycles"] + values["renderer_cycles"] + 25
        )
        verify_memory_path(
            hello[0], message, cache_bypass=False, warm_slc=index != 0
        )

    expected_names = {
        "fill_solid_sample_000001.png",
        "fill_solid_sample_000002.png",
    }
    actual_names = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*.png")
    }
    assert actual_names == expected_names, (
        f"fill_solid: PNG artifacts {sorted(actual_names)!r}, "
        f"expected {sorted(expected_names)!r}"
    )
    part_files = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name.endswith(".part")
    ]
    assert not part_files, f"fill_solid: incomplete artifacts remain: {part_files}"

    expected_pixels = b"\xFF\x00\x00\xFF" * PIXEL_COUNT
    for name in sorted(expected_names):
        width, height, pixels = decode_rgba8_png(output_dir / name)
        assert (width, height) == (WIDTH, HEIGHT), (
            f"{name}: got {width}x{height}, expected {WIDTH}x{HEIGHT}"
        )
        assert pixels == expected_pixels, f"{name}: framebuffer is not solid red"


def verify_fill_depth_case(
    executable: Path,
    case_name: str,
    output_dir: Path,
    *,
    expected_ps_invocations: int,
    expected_depth_rejected: int,
    expected_pixel: bytes,
    cache_bypass: str,
) -> None:
    completed = invoke_case(
        executable,
        case_name,
        output_dir,
        frames=1,
        cache_bypass=cache_bypass,
    )
    assert completed.returncode == 0, (
        f"{case_name} model run failed:\n" + process_details(completed)
    )

    messages = json_messages(completed)
    hello = [message for message in messages if message.get("type") == "hello"]
    counters = [
        message for message in messages if message.get("type") == "counter"
    ]
    done = [message for message in messages if message.get("type") == "done"]
    assert len(hello) == 1, f"{case_name}: missing/duplicate hello"
    assert hello[0].get("functional_scope") == f"{case_name}-pco-iss-v1"
    assert hello[0].get("shader_binary") == "mesa-pco-public-encoding"
    assert len(counters) == 1, f"{case_name}: missing/duplicate counter"
    verify_memory_path(
        hello[0], counters[0], cache_bypass=cache_bypass == "on"
    )
    assert len(done) == 1 and done[0].get("pool_leaks") == 0, (
        f"{case_name}: missing done or MemoryPool leak"
    )

    message = counters[0]
    values = message.get("counters")
    assert isinstance(values, dict), f"{case_name}: counters is not an object"
    expected_counters = {
        "ia_vertices": 4,
        "ia_primitives": 2,
        "vs_invocations": 4,
        "c_invocations": 2,
        "c_primitives": 2,
        "ps_invocations": expected_ps_invocations,
        "drawlists": 1,
        "setup_triangles": 2,
        "tiles_binned": 4,
        "tiles_scheduled": 4,
        "covered_pixels": PIXEL_COUNT,
        "fragment_candidates": PIXEL_COUNT,
        "hsr_rejected_fragments": expected_depth_rejected,
        "depth_tested_fragments": PIXEL_COUNT,
        "depth_rejected_fragments": expected_depth_rejected,
        "depth_written_fragments": (
            expected_ps_invocations
            if expected_depth_rejected == 0
            else 0
        ),
        "pco_instructions": 6,
        "vs_alu_instructions": 0,
        "vs_tex_instructions": 0,
        "vs_memory_instructions": 16,
        "fs_alu_instructions": expected_ps_invocations * 4,
        "fs_tex_instructions": 0,
        "fs_memory_instructions": 0,
        "usc_groups": (
            1
            if expected_ps_invocations == 0
            else USC_GROUP_COUNT
        ),
        "texture_requests": 0,
        "texel_fetches": 0,
        "pbe_color_reads": 0,
        "pbe_blended_fragments": 0,
        "pbe_fragment_writes": expected_ps_invocations,
        "pbe_pixels_written": PIXEL_COUNT,
    }
    for field, expected in expected_counters.items():
        if field in UNIFIED_MEMORY_EXACT_SKIP:
            continue
        assert values.get(field) == expected, (
            f"{case_name}: {field}={values.get(field)!r}, expected {expected}"
        )
    assert values["fragment_candidates"] == (
        values["ps_invocations"] + values["hsr_rejected_fragments"]
    ), f"{case_name}: ISP candidate conservation failed"
    assert message.get("drawlist_stats") == [
        {
            "drawlist": 0,
            "draw_id": 0,
            "vs": {
                "invocations": 4,
                "program": {
                    "groups": 2,
                    "instructions": 2,
                    "alu": 0,
                    "tex": 0,
                    "memory": 2,
                },
                "executed": {"alu": 0, "tex": 0, "memory": 16},
            },
            "fs": {
                "invocations": expected_ps_invocations,
                "program": {
                    "groups": 4,
                    "instructions": 4,
                    "alu": 4,
                    "tex": 0,
                    "memory": 0,
                },
                "executed": {
                    "alu": expected_ps_invocations * 4,
                    "tex": 0,
                    "memory": 0,
                },
            },
        }
    ], f"{case_name}: per-DrawList statistics mismatch"

    artifact = output_dir / f"{case_name}_sample_000001.png"
    width, height, pixels = decode_rgba8_png(artifact)
    assert (width, height) == (WIDTH, HEIGHT), f"{case_name}: PNG size mismatch"
    assert pixels == expected_pixel * PIXEL_COUNT, (
        f"{case_name}: framebuffer pixels do not match the llvmpipe golden color"
    )


def verify_fill_solid_blended(executable: Path, output_dir: Path) -> None:
    case_name = "fill_solid_blended"
    completed = invoke_case(executable, case_name, output_dir, frames=1)
    assert completed.returncode == 0, (
        f"{case_name} model run failed:\n" + process_details(completed)
    )

    messages = json_messages(completed)
    hello = [message for message in messages if message.get("type") == "hello"]
    counters = [
        message for message in messages if message.get("type") == "counter"
    ]
    done = [message for message in messages if message.get("type") == "done"]
    assert len(hello) == 1 and len(counters) == 1
    assert len(done) == 1 and done[0].get("pool_leaks") == 0
    assert hello[0].get("mode") == "systemc-functional-fill-blended"
    assert hello[0].get("functional_scope") == f"{case_name}-pco-iss-v1"
    assert hello[0].get("effective_early_hsr") is False
    assert hello[0].get("blend_state") == {
        "enabled": True,
        "rgb_equation": "add",
        "alpha_equation": "add",
        "source_rgb_factor": "source-alpha",
        "destination_rgb_factor": "one-minus-source-alpha",
        "source_alpha_factor": "source-alpha",
        "destination_alpha_factor": "one-minus-source-alpha",
    }
    verify_memory_path(hello[0], counters[0], cache_bypass=False)

    values = counters[0].get("counters")
    assert isinstance(values, dict)
    expected_counters = {
        "ia_vertices": 4,
        "ia_primitives": 2,
        "vs_invocations": 4,
        "c_invocations": 2,
        "c_primitives": 2,
        "ps_invocations": PIXEL_COUNT,
        "drawlists": 1,
        "setup_triangles": 2,
        "texel_fetches": 0,
        "texture_requests": 0,
        "tiles_binned": 4,
        "tiles_scheduled": 4,
        "covered_pixels": PIXEL_COUNT,
        "fragment_candidates": PIXEL_COUNT,
        "hsr_rejected_fragments": 0,
        "depth_tested_fragments": 0,
        "depth_rejected_fragments": 0,
        "depth_written_fragments": 0,
        "pco_instructions": 6,
        "vs_alu_instructions": 0,
        "vs_tex_instructions": 0,
        "vs_memory_instructions": 16,
        "fs_alu_instructions": PIXEL_COUNT * 4,
        "fs_tex_instructions": 0,
        "fs_memory_instructions": 0,
        "usc_groups": USC_GROUP_COUNT,
        "pbe_color_reads": PIXEL_COUNT,
        "pbe_blended_fragments": PIXEL_COUNT,
        "pbe_fragment_writes": PIXEL_COUNT,
        "pbe_pixels_written": PIXEL_COUNT,
    }
    for field, expected in expected_counters.items():
        if field in UNIFIED_MEMORY_EXACT_SKIP:
            continue
        assert values.get(field) == expected, (
            f"{case_name}: {field}={values.get(field)!r}, expected {expected}"
        )
    assert values["fragment_candidates"] == (
        values["ps_invocations"] + values["hsr_rejected_fragments"]
    )
    assert counters[0].get("drawlist_stats") == [
        {
            "drawlist": 0,
            "draw_id": 0,
            "vs": {
                "invocations": 4,
                "program": {
                    "groups": 2,
                    "instructions": 2,
                    "alu": 0,
                    "tex": 0,
                    "memory": 2,
                },
                "executed": {"alu": 0, "tex": 0, "memory": 16},
            },
            "fs": {
                "invocations": PIXEL_COUNT,
                "program": {
                    "groups": 4,
                    "instructions": 4,
                    "alu": 4,
                    "tex": 0,
                    "memory": 0,
                },
                "executed": {
                    "alu": PIXEL_COUNT * 4,
                    "tex": 0,
                    "memory": 0,
                },
            },
        }
    ]
    artifact = output_dir / f"{case_name}_sample_000001.png"
    width, height, pixels = decode_rgba8_png(artifact)
    assert (width, height) == (WIDTH, HEIGHT)
    assert pixels == b"\xFF\x00\x00\xFF" * PIXEL_COUNT


def verify_fill_solid_bypass(
    executable: Path, output_dir: Path, golden_pixels: bytes
) -> None:
    completed = invoke_case(
        executable,
        "fill_solid",
        output_dir,
        frames=1,
        cache_bypass="on",
    )
    assert completed.returncode == 0, (
        "fill_solid bypass run failed:\n" + process_details(completed)
    )
    messages = json_messages(completed)
    hello = [message for message in messages if message.get("type") == "hello"]
    counters = [
        message for message in messages if message.get("type") == "counter"
    ]
    done = [message for message in messages if message.get("type") == "done"]
    assert len(hello) == 1 and len(counters) == 1
    assert len(done) == 1 and done[0].get("pool_leaks") == 0
    verify_memory_path(hello[0], counters[0], cache_bypass=True)
    _, _, pixels = decode_rgba8_png(
        output_dir / "fill_solid_sample_000001.png"
    )
    assert pixels == golden_pixels, (
        "fill_solid cache-on and cache-bypass PNG bytes differ"
    )


def verify_fill_solid_direct(
    executable: Path, output_dir: Path, golden_pixels: bytes
) -> None:
    completed = invoke_case(
        executable,
        "fill_solid",
        output_dir,
        frames=1,
        memory_mode="direct",
    )
    assert completed.returncode == 0, (
        "fill_solid direct run failed:\n" + process_details(completed)
    )
    messages = json_messages(completed)
    hello = [message for message in messages if message.get("type") == "hello"]
    counters = [
        message for message in messages if message.get("type") == "counter"
    ]
    done = [message for message in messages if message.get("type") == "done"]
    assert len(hello) == 1 and len(counters) == 1
    assert len(done) == 1 and done[0].get("pool_leaks") == 0
    assert hello[0].get("cache_bypass") is False
    assert hello[0].get("memory_mode") == "direct"
    assert hello[0].get("cache_simulated") is False
    values = counters[0].get("counters")
    assert isinstance(values, dict)
    assert values["pixel_data_master_transactions"] == 1
    assert values["pixel_data_master_bytes"] == FRAMEBUFFER_BYTES
    assert values["framebuffer_dram_readback_bytes"] == FRAMEBUFFER_BYTES
    for field in (
        "tcu_line_accesses",
        "tcu_read_accesses",
        "tcu_hits",
        "tcu_misses",
        "tcu_evictions",
        "tcu_writebacks",
        "tcu_bypassed",
        "tcu_cycles",
        "slc_line_accesses",
        "slc_read_accesses",
        "slc_write_accesses",
        "slc_hits",
        "slc_misses",
        "slc_evictions",
        "slc_writebacks",
        "slc_bypassed",
        "slc_cycles",
        "dram_read_transactions",
        "dram_write_transactions",
        "dram_read_bytes",
        "dram_write_bytes",
        "dram_cycles",
    ):
        assert values[field] == 0, (
            f"direct memory path: {field}={values[field]!r}, expected 0"
        )
    assert values["memory_direct_read_bytes"] >= FRAMEBUFFER_BYTES
    assert values["memory_direct_write_bytes"] >= FRAMEBUFFER_BYTES
    _, _, pixels = decode_rgba8_png(
        output_dir / "fill_solid_sample_000001.png"
    )
    assert pixels == golden_pixels, (
        "fill_solid cache-on and direct PNG bytes differ"
    )


def verify_triangle_setup(executable: Path, output_dir: Path) -> None:
    case_name = "triangle_setup"
    width = 64
    height = 64
    pixel_count = width * height
    framebuffer_bytes = pixel_count * 4
    slc_lines = framebuffer_bytes // SLC_LINE_BYTES
    completed = invoke_case(
        executable,
        case_name,
        output_dir,
        frames=1,
        width=width,
        height=height,
        timeout=60,
    )
    assert completed.returncode == 0, (
        f"{case_name} model run failed:\n" + process_details(completed)
    )

    messages = json_messages(completed)
    hello = [message for message in messages if message.get("type") == "hello"]
    counters = [
        message for message in messages if message.get("type") == "counter"
    ]
    done = [message for message in messages if message.get("type") == "done"]
    assert len(hello) == 1 and len(counters) == 1
    assert len(done) == 1 and done[0].get("pool_leaks") == 0
    assert done[0].get("pool_bytes_in_flight") == 0
    assert hello[0].get("mode") == "systemc-functional-triangle-setup"
    assert hello[0].get("functional_scope") == f"{case_name}-pco-iss-v1"
    assert hello[0].get("shader_binary") == "mesa-pco-public-encoding"
    assert hello[0].get("effective_early_hsr") is True
    assert hello[0].get("blend_state") == {
        "enabled": False,
        "rgb_equation": "add",
        "alpha_equation": "add",
        "source_rgb_factor": "one",
        "destination_rgb_factor": "zero",
        "source_alpha_factor": "one",
        "destination_alpha_factor": "zero",
    }

    message = counters[0]
    values = message.get("counters")
    assert isinstance(values, dict)
    expected_counters = {
        "ia_vertices": 98304,
        "ia_primitives": 32768,
        "vs_invocations": 21144,
        "c_invocations": 32768,
        "c_primitives": 8970,
        "ps_invocations": pixel_count,
        "drawlists": 1,
        "setup_triangles": 8970,
        "texel_fetches": 0,
        "texture_requests": 0,
        "tiles_binned": 4,
        "tiles_scheduled": 4,
        "covered_pixels": pixel_count,
        "fragment_candidates": pixel_count,
        "hsr_rejected_fragments": 0,
        "depth_tested_fragments": 0,
        "depth_rejected_fragments": 0,
        "depth_written_fragments": 0,
        "pco_instructions": 6,
        "vs_alu_instructions": 0,
        "vs_tex_instructions": 0,
        "vs_memory_instructions": 21144 * 4,
        "fs_alu_instructions": pixel_count * 4,
        "fs_tex_instructions": 0,
        "fs_memory_instructions": 0,
        "usc_groups": 9382,
        "pbe_color_reads": 0,
        "pbe_blended_fragments": 0,
        "pbe_fragment_writes": pixel_count,
        "pbe_pixels_written": pixel_count,
        "pixel_data_master_transactions": 1,
        "pixel_data_master_bytes": framebuffer_bytes,
        "slc_line_accesses": slc_lines,
        "slc_write_accesses": slc_lines,
        "slc_misses": slc_lines,
        "slc_writebacks": slc_lines,
        "dram_read_transactions": 1,
        "dram_write_transactions": slc_lines,
        "dram_read_bytes": framebuffer_bytes,
        "dram_write_bytes": framebuffer_bytes,
        "framebuffer_dram_readback_bytes": framebuffer_bytes,
    }
    for field, expected in expected_counters.items():
        if field in UNIFIED_MEMORY_EXACT_SKIP:
            continue
        assert values.get(field) == expected, (
            f"{case_name}: {field}={values.get(field)!r}, expected {expected}"
        )
    assert values["fragment_candidates"] == (
        values["ps_invocations"] + values["hsr_rejected_fragments"]
    )
    for field in STAGE_COUNTERS:
        value = values.get(field)
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        assert value > 0, f"{case_name}: {field} not active"
    assert values["virtual_gpu_cycles"] == (
        values["tiler_cycles"] + values["renderer_cycles"] + 25
    )
    verify_memory_path(
        hello[0],
        message,
        cache_bypass=False,
        framebuffer_bytes=framebuffer_bytes,
    )
    assert message.get("drawlist_stats") == [
        {
            "drawlist": 0,
            "draw_id": 0,
            "vs": {
                "invocations": 21144,
                "program": {
                    "groups": 2,
                    "instructions": 2,
                    "alu": 0,
                    "tex": 0,
                    "memory": 2,
                },
                "executed": {"alu": 0, "tex": 0, "memory": 21144 * 4},
            },
            "fs": {
                "invocations": pixel_count,
                "program": {
                    "groups": 4,
                    "instructions": 4,
                    "alu": 4,
                    "tex": 0,
                    "memory": 0,
                },
                "executed": {
                    "alu": pixel_count * 4,
                    "tex": 0,
                    "memory": 0,
                },
            },
        }
    ]
    artifact = output_dir / f"{case_name}_sample_000001.png"
    png_width, png_height, pixels = decode_rgba8_png(artifact)
    assert (png_width, png_height) == (width, height)
    assert pixels == b"\xFF\x80\x00\xFF" * pixel_count


def verify_triangle_setup_all_culled(
    executable: Path, output_dir: Path
) -> None:
    case_name = "triangle_setup_all_culled"
    width = 64
    height = 64
    pixel_count = width * height
    framebuffer_bytes = pixel_count * 4
    completed = invoke_case(
        executable,
        case_name,
        output_dir,
        frames=1,
        width=width,
        height=height,
        timeout=60,
    )
    assert completed.returncode == 0, (
        f"{case_name} model run failed:\n" + process_details(completed)
    )

    messages = json_messages(completed)
    hello = [message for message in messages if message.get("type") == "hello"]
    counters = [
        message for message in messages if message.get("type") == "counter"
    ]
    done = [message for message in messages if message.get("type") == "done"]
    assert len(hello) == 1 and len(counters) == 1
    assert len(done) == 1 and done[0].get("pool_leaks") == 0
    assert done[0].get("pool_bytes_in_flight") == 0
    assert hello[0].get("mode") == (
        "systemc-functional-triangle-setup-all-culled"
    )
    assert hello[0].get("functional_scope") == f"{case_name}-pco-iss-v1"
    assert hello[0].get("pco_subset") == "mbyp-uvsw-triangle-setup"
    assert hello[0].get("effective_early_hsr") is True
    assert hello[0].get("face_cull_state") == {
        "enabled": True,
        "mode": "back",
        "front_face": "counter-clockwise",
    }

    message = counters[0]
    values = message.get("counters")
    assert isinstance(values, dict)
    expected_counters = {
        "ia_vertices": 98304,
        "ia_primitives": 32768,
        "vs_invocations": 21144,
        "c_invocations": 32768,
        "c_primitives": 5456,
        "ps_invocations": 0,
        "drawlists": 1,
        "setup_triangles": 5456,
        "texel_fetches": 0,
        "texture_requests": 0,
        "tiles_binned": 4,
        "tiles_scheduled": 4,
        "covered_pixels": 0,
        "fragment_candidates": 0,
        "hsr_rejected_fragments": 0,
        "depth_tested_fragments": 0,
        "depth_rejected_fragments": 0,
        "depth_written_fragments": 0,
        "pco_instructions": 6,
        "vs_alu_instructions": 0,
        "vs_tex_instructions": 0,
        "vs_memory_instructions": 21144 * 4,
        "fs_alu_instructions": 0,
        "fs_tex_instructions": 0,
        "fs_memory_instructions": 0,
        "usc_groups": 5286,
        "texture_cycles": 0,
        "pbe_color_reads": 0,
        "pbe_blended_fragments": 0,
        "pbe_fragment_writes": 0,
        # PBE publishes the complete clear-initialized RGBA8 surface even
        # when face culling leaves no fragment color writes.
        "pbe_pixels_written": pixel_count,
    }
    for field, expected in expected_counters.items():
        if field in UNIFIED_MEMORY_EXACT_SKIP:
            continue
        assert values.get(field) == expected, (
            f"{case_name}: {field}={values.get(field)!r}, expected {expected}"
        )
    assert values["fragment_candidates"] == (
        values["ps_invocations"] + values["hsr_rejected_fragments"]
    )
    for field in STAGE_COUNTERS:
        if field == "texture_cycles":
            continue
        value = values.get(field)
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        assert value > 0, f"{case_name}: {field} not active"
    assert values["virtual_gpu_cycles"] == (
        values["tiler_cycles"] + values["renderer_cycles"] + 25
    )
    verify_memory_path(
        hello[0],
        message,
        cache_bypass=False,
        framebuffer_bytes=framebuffer_bytes,
    )
    assert message.get("drawlist_stats") == [
        {
            "drawlist": 0,
            "draw_id": 0,
            "vs": {
                "invocations": 21144,
                "program": {
                    "groups": 2,
                    "instructions": 2,
                    "alu": 0,
                    "tex": 0,
                    "memory": 2,
                },
                "executed": {"alu": 0, "tex": 0, "memory": 21144 * 4},
            },
            "fs": {
                "invocations": 0,
                "program": {
                    "groups": 4,
                    "instructions": 4,
                    "alu": 4,
                    "tex": 0,
                    "memory": 0,
                },
                "executed": {"alu": 0, "tex": 0, "memory": 0},
            },
        }
    ]
    artifact = output_dir / f"{case_name}_sample_000001.png"
    png_width, png_height, pixels = decode_rgba8_png(artifact)
    assert (png_width, png_height) == (width, height)
    assert pixels == b"\x00\xFF\x00\xFF" * pixel_count


def verify_triangle_setup_half_culled(
    executable: Path, output_dir: Path
) -> None:
    case_name = "triangle_setup_half_culled"
    width = 64
    height = 64
    pixel_count = width * height
    framebuffer_bytes = pixel_count * 4
    completed = invoke_case(
        executable,
        case_name,
        output_dir,
        frames=1,
        width=width,
        height=height,
        timeout=60,
    )
    assert completed.returncode == 0, (
        f"{case_name} model run failed:\n" + process_details(completed)
    )

    messages = json_messages(completed)
    hello = [message for message in messages if message.get("type") == "hello"]
    counters = [
        message for message in messages if message.get("type") == "counter"
    ]
    done = [message for message in messages if message.get("type") == "done"]
    assert len(hello) == 1 and len(counters) == 1
    assert len(done) == 1 and done[0].get("pool_leaks") == 0
    assert done[0].get("pool_bytes_in_flight") == 0
    assert hello[0].get("mode") == (
        "systemc-functional-triangle-setup-half-culled"
    )
    assert hello[0].get("functional_scope") == f"{case_name}-pco-iss-v1"
    assert hello[0].get("shader_binary") == "mesa-pco-public-encoding"
    assert hello[0].get("pco_subset") == "mbyp-uvsw-triangle-setup"
    assert hello[0].get("effective_early_hsr") is True
    assert hello[0].get("face_cull_state") == {
        "enabled": True,
        "mode": "back",
        "front_face": "counter-clockwise",
    }

    message = counters[0]
    values = message.get("counters")
    assert isinstance(values, dict)
    expected_counters = {
        "ia_vertices": 98304,
        "ia_primitives": 32768,
        "vs_invocations": 21144,
        "c_invocations": 32768,
        "c_primitives": 6797,
        "ps_invocations": 2044,
        "drawlists": 1,
        "setup_triangles": 6797,
        "texel_fetches": 0,
        "texture_requests": 0,
        "tiles_binned": 4,
        "tiles_scheduled": 4,
        "covered_pixels": 2044,
        "fragment_candidates": 2044,
        "hsr_rejected_fragments": 0,
        "depth_tested_fragments": 0,
        "depth_rejected_fragments": 0,
        "depth_written_fragments": 0,
        "pco_instructions": 6,
        "vs_alu_instructions": 0,
        "vs_tex_instructions": 0,
        "vs_memory_instructions": 21144 * 4,
        "fs_alu_instructions": 2044 * 4,
        "fs_tex_instructions": 0,
        "fs_memory_instructions": 0,
        "usc_groups": 5286 + 2044,
        "pbe_color_reads": 0,
        "pbe_blended_fragments": 0,
        "pbe_fragment_writes": 2044,
        "pbe_pixels_written": pixel_count,
    }
    for field, expected in expected_counters.items():
        if field in UNIFIED_MEMORY_EXACT_SKIP:
            continue
        assert values.get(field) == expected, (
            f"{case_name}: {field}={values.get(field)!r}, expected {expected}"
        )
    assert values["fragment_candidates"] == (
        values["ps_invocations"] + values["hsr_rejected_fragments"]
    )
    for field in STAGE_COUNTERS:
        value = values.get(field)
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        assert value > 0, f"{case_name}: {field} not active"
    assert values["virtual_gpu_cycles"] == (
        values["tiler_cycles"] + values["renderer_cycles"] + 25
    )
    verify_memory_path(
        hello[0],
        message,
        cache_bypass=False,
        framebuffer_bytes=framebuffer_bytes,
    )
    assert message.get("drawlist_stats") == [
        {
            "drawlist": 0,
            "draw_id": 0,
            "vs": {
                "invocations": 21144,
                "program": {
                    "groups": 2,
                    "instructions": 2,
                    "alu": 0,
                    "tex": 0,
                    "memory": 2,
                },
                "executed": {"alu": 0, "tex": 0, "memory": 21144 * 4},
            },
            "fs": {
                "invocations": 2044,
                "program": {
                    "groups": 4,
                    "instructions": 4,
                    "alu": 4,
                    "tex": 0,
                    "memory": 0,
                },
                "executed": {"alu": 2044 * 4, "tex": 0, "memory": 0},
            },
        }
    ]
    artifact = output_dir / f"{case_name}_sample_000001.png"
    png_width, png_height, pixels = decode_rgba8_png(artifact)
    assert (png_width, png_height) == (width, height)
    assert pixels == triangle_setup_half_culled_golden_pixels()


def verify_attribute_fetch_shader(
    executable: Path,
    output_dir: Path,
    case_name: str = "attribute_fetch_shader",
) -> None:
    attribute_counts = {
        "attribute_fetch_shader": 1,
        "attribute_fetch_shader_2_attr": 2,
        "attribute_fetch_shader_4_attr": 4,
        "attribute_fetch_shader_8_attr": 8,
    }
    assert case_name in attribute_counts
    attribute_count = attribute_counts[case_name]
    width = 64
    height = 64
    pixel_count = width * height
    framebuffer_bytes = pixel_count * 4
    completed = invoke_case(
        executable,
        case_name,
        output_dir,
        frames=1,
        width=width,
        height=height,
        timeout=60,
    )
    assert completed.returncode == 0, (
        f"{case_name} model run failed:\n" + process_details(completed)
    )

    messages = json_messages(completed)
    hello = [message for message in messages if message.get("type") == "hello"]
    counters = [
        message for message in messages if message.get("type") == "counter"
    ]
    done = [message for message in messages if message.get("type") == "done"]
    assert len(hello) == 1 and len(counters) == 1
    assert len(done) == 1 and done[0].get("pool_leaks") == 0
    assert done[0].get("pool_bytes_in_flight") == 0
    assert hello[0].get("mode") == (
        "systemc-functional-attribute-fetch-shader"
    )
    assert hello[0].get("functional_scope") == f"{case_name}-pco-iss-v1"
    assert hello[0].get("shader_binary") == "mesa-pco-public-encoding"
    assert hello[0].get("pco_subset") == (
        "fadd-mbyp-uvsw-attribute-fetch"
        if attribute_count > 1
        else "mbyp-uvsw-attribute-fetch"
    )
    assert hello[0].get("effective_early_hsr") is True
    assert hello[0].get("face_cull_state") == {
        "enabled": True,
        "mode": "back",
        "front_face": "counter-clockwise",
    }

    message = counters[0]
    expected_vertex_pco_binary = {
        1: {"fingerprint": "fnv1a64:48cf8717db7aa8cf", "bytes": 56},
        2: {"fingerprint": "fnv1a64:4fb7f3aba4b44c19", "bytes": 56},
        4: {"fingerprint": "fnv1a64:c54ea51cdaab08a0", "bytes": 96},
        8: {"fingerprint": "fnv1a64:87d4d7b5e46ff241", "bytes": 176},
    }[attribute_count]
    expected_vertex_pco_opcodes = {
        "fadd": {1: 0, 2: 2, 4: 6, 8: 14}[attribute_count],
        "mbyp": 4 if attribute_count == 1 else 2,
        "uvsw_write": 1,
        "uvsw_write_emit_endtask": 0,
        "uvsw_emit_endtask": 1,
    }
    assert message.get("vertex_pco_binary") == expected_vertex_pco_binary
    assert message.get("vertex_pco_opcodes") == expected_vertex_pco_opcodes
    values = message.get("counters")
    assert isinstance(values, dict)
    expected_counters = {
        "ia_vertices": 24576,
        "ia_primitives": 8192,
        "vs_invocations": 5317,
        "c_invocations": 8192,
        "c_primitives": 8192,
        "ps_invocations": 0,
        "drawlists": 1,
        "setup_triangles": 8192,
        "texel_fetches": 0,
        "texture_requests": 0,
        "tiles_binned": 4,
        "tiles_scheduled": 4,
        "covered_pixels": 0,
        "fragment_candidates": 0,
        "hsr_rejected_fragments": 0,
        "depth_tested_fragments": 0,
        "depth_rejected_fragments": 0,
        "depth_written_fragments": 0,
        "vertex_attribute_fetches": 5317 * attribute_count,
        "vertex_attribute_bytes": 5317 * attribute_count * 2 * 4,
        "pco_instructions": {1: 10, 2: 10, 4: 14, 8: 22}[attribute_count],
        "vs_alu_instructions": 5317 * {1: 4, 2: 4, 4: 8, 8: 16}[
            attribute_count
        ],
        "vs_tex_instructions": 0,
        "vs_memory_instructions": 5317 * 5,
        "fs_alu_instructions": 0,
        "fs_tex_instructions": 0,
        "fs_memory_instructions": 0,
        "usc_groups": 1330,
        "pbe_color_reads": 0,
        "pbe_blended_fragments": 0,
        "pbe_fragment_writes": 0,
        "pbe_pixels_written": pixel_count,
        "tiler_cycles": {1: 1367, 2: 1388, 4: 1431, 8: 1516}[
            attribute_count
        ],
        "renderer_cycles": 330,
        "virtual_gpu_cycles": {1: 1722, 2: 1743, 4: 1786, 8: 1871}[
            attribute_count
        ],
    }
    for field, expected in expected_counters.items():
        if field in UNIFIED_MEMORY_EXACT_SKIP:
            continue
        assert values.get(field) == expected, (
            f"{case_name}: {field}={values.get(field)!r}, expected {expected}"
        )
    expected_stage_cycles = {
        "vdm_cycles": 56,
        "vertex_fetch_cycles": {1: 25, 2: 46, 4: 88, 8: 171}[
            attribute_count
        ],
        "pco_decode_cycles": {1: 9, 2: 9, 4: 10, 8: 12}[
            attribute_count
        ],
        "usc_slot_cycles": 667,
        "usc_cluster_cycles": 337,
        "clip_cull_cycles": 70,
        "tiler_bin_cycles": 138,
        "parameter_buffer_cycles": 69,
        "tile_scheduler_cycles": 9,
        "isp_cycles": 10,
        "fragment_frontend_cycles": 5,
        "texture_cycles": 0,
        "pbe_cycles": 44,
        "pixel_data_master_cycles": 1,
        "slc_cycles": 128,
        "dram_cycles": 129,
    }
    for field, expected in expected_stage_cycles.items():
        if field in UNIFIED_MEMORY_EXACT_SKIP:
            continue
        assert values.get(field) == expected, (
            f"{case_name}: {field}={values.get(field)!r}, expected {expected}"
        )
    assert values["fragment_candidates"] == (
        values["ps_invocations"] + values["hsr_rejected_fragments"]
    )
    assert values["virtual_gpu_cycles"] == (
        values["tiler_cycles"] + values["renderer_cycles"] + 25
    )
    verify_memory_path(
        hello[0],
        message,
        cache_bypass=False,
        framebuffer_bytes=framebuffer_bytes,
    )
    assert message.get("drawlist_stats") == [
        {
            "drawlist": 0,
            "draw_id": 0,
            "vs": {
                "invocations": 5317,
                "program": {
                    "groups": {1: 6, 2: 6, 4: 10, 8: 18}[
                        attribute_count
                    ],
                    "instructions": {1: 6, 2: 6, 4: 10, 8: 18}[
                        attribute_count
                    ],
                    "alu": {1: 4, 2: 4, 4: 8, 8: 16}[attribute_count],
                    "tex": 0,
                    "memory": 2,
                },
                "executed": {
                    "alu": 5317 * {1: 4, 2: 4, 4: 8, 8: 16}[
                        attribute_count
                    ],
                    "tex": 0,
                    "memory": 5317 * 5,
                },
            },
            "fs": {
                "invocations": 0,
                "program": {
                    "groups": 4,
                    "instructions": 4,
                    "alu": 4,
                    "tex": 0,
                    "memory": 0,
                },
                "executed": {"alu": 0, "tex": 0, "memory": 0},
            },
        }
    ]
    artifact = output_dir / f"{case_name}_sample_000001.png"
    png_width, png_height, pixels = decode_rgba8_png(artifact)
    assert (png_width, png_height) == (width, height)
    assert pixels == b"\x00\xFF\x00\xFF" * pixel_count


def verify_varyings_shader_one(executable: Path, output_dir: Path) -> None:
    completed = invoke_case(
        executable,
        "varyings_shader_1",
        output_dir,
        frames=1,
        cache_bypass="off",
        width=64,
        height=64,
    )
    assert completed.returncode == 0, (
        "varyings_shader_1 model run failed:\n" + process_details(completed)
    )
    messages = json_messages(completed)
    hello = [message for message in messages if message.get("type") == "hello"]
    counters = [
        message for message in messages if message.get("type") == "counter"
    ]
    done = [message for message in messages if message.get("type") == "done"]
    assert len(hello) == len(counters) == len(done) == 1
    assert hello[0].get("mode") == "systemc-functional-varyings-shader"
    assert hello[0].get("functional_scope") == "varyings_shader_1-pco-iss-v1"
    assert hello[0].get("pco_subset") == "fitrp-wdf-mbyp-uvsw-varying"
    assert hello[0].get("shader_binary") == "mesa-pco-public-encoding"
    assert hello[0].get("cache_bypass") is False
    assert hello[0].get("face_cull_state") == {
        "enabled": False,
        "mode": "back",
        "front_face": "counter-clockwise",
    }
    assert done[0].get("pool_leaks") == 0
    assert done[0].get("pool_bytes_in_flight") == 0

    message = counters[0]
    assert message.get("provenance") == "modeled"
    assert message.get("vertex_pco_binary") == {
        "fingerprint": "fnv1a64:c1a8f8a4f58fc81f",
        "bytes": 72,
    }
    assert message.get("vertex_pco_opcodes") == {
        "fadd": 0,
        "mbyp": 4,
        "uvsw_write": 2,
        "uvsw_write_emit_endtask": 0,
        "uvsw_emit_endtask": 1,
    }
    assert message.get("fragment_pco_binary") == {
        "fingerprint": "fnv1a64:76458bbcec6f53bf",
        "bytes": 48,
    }
    assert message.get("fragment_pco_opcodes") == {
        "fitrp": 1,
        "wdf": 1,
        "mbyp": 4,
    }
    values = message.get("counters")
    assert isinstance(values, dict)
    expected = {
        "ia_vertices": 96,
        "ia_primitives": 32,
        "vs_invocations": 25,
        "gs_invocations": 0,
        "gs_primitives": 0,
        "c_invocations": 32,
        "c_primitives": 32,
        "ps_invocations": 4096,
        "hs_invocations": 0,
        "ds_invocations": 0,
        "cs_invocations": 0,
        "ts_invocations": 0,
        "ms_invocations": 0,
        "ms_primitives": 0,
        "drawlists": 1,
        "setup_triangles": 32,
        "texel_fetches": 0,
        "vertex_attribute_fetches": 25,
        "vertex_attribute_bytes": 200,
        "pco_decode_cycles": 10,
        "pco_instructions": 13,
        "vs_alu_instructions": 100,
        "vs_tex_instructions": 0,
        "vs_memory_instructions": 225,
        "fs_alu_instructions": 20480,
        "fs_tex_instructions": 0,
        "fs_memory_instructions": 0,
        "fifo_stall_events": 0,
        "pool_bytes_in_flight": 0,
        "vdm_cycles": 9,
        "vertex_fetch_cycles": 5,
        "clip_cull_cycles": 7,
        "tiler_bin_cycles": 11,
        "parameter_buffer_cycles": 6,
        "parameter_coefficient_sets": 160,
        "parameter_write_bytes": 2560,
        "pds_coefficient_tasks": 1152,
        "pds_douti_issues": 2304,
        "usc_coefficient_load_bytes": 92160,
        "tile_scheduler_cycles": 9,
        "isp_cycles": 74,
        "fragment_frontend_cycles": 37,
        "texture_cycles": 2,
        "pbe_cycles": 44,
        "texture_requests": 0,
        "tiles_binned": 4,
        "tiles_scheduled": 4,
        "covered_pixels": 4096,
        "fragment_candidates": 4096,
        "hsr_rejected_fragments": 0,
        "depth_tested_fragments": 0,
        "depth_rejected_fragments": 0,
        "depth_written_fragments": 0,
        "pbe_color_reads": 0,
        "pbe_blended_fragments": 0,
        "pbe_fragment_writes": 4096,
        "pbe_pixels_written": 4096,
        "virtual_gpu_cycles": 1379,
        "tiler_cycles": 55,
        "renderer_cycles": 1299,
        "usc_groups": 1159,
        "usc_slot_cycles": 584,
        "usc_cluster_cycles": 298,
    }
    for field, wanted in expected.items():
        if field in UNIFIED_MEMORY_EXACT_SKIP:
            continue
        assert values.get(field) == wanted, (
            f"varyings_shader_1: {field}={values.get(field)!r}, expected {wanted}"
        )
    assert int(values.get("pool_high_water_bytes", 0)) > 0
    assert values["parameter_coefficient_sets"] == 32 * 5
    assert values["parameter_write_bytes"] == (
        values["parameter_coefficient_sets"] * 16
    )
    assert values["pds_coefficient_tasks"] == 1152
    assert values["pds_douti_issues"] == values["pds_coefficient_tasks"] * 2
    assert values["usc_coefficient_load_bytes"] == (
        values["pds_coefficient_tasks"] * 20 * 4
    )
    assert values["fragment_candidates"] == (
        values["ps_invocations"] + values["hsr_rejected_fragments"]
    )
    assert values["virtual_gpu_cycles"] == (
        values["tiler_cycles"] + values["renderer_cycles"] + 25
    )
    verify_memory_path(
        hello[0],
        message,
        cache_bypass=False,
        framebuffer_bytes=64 * 64 * 4,
    )
    assert message.get("drawlist_stats") == [
        {
            "drawlist": 0,
            "draw_id": 0,
            "vs": {
                "invocations": 25,
                "program": {
                    "groups": 7,
                    "instructions": 7,
                    "alu": 4,
                    "tex": 0,
                    "memory": 3,
                },
                "executed": {"alu": 100, "tex": 0, "memory": 225},
            },
            "fs": {
                "invocations": 4096,
                "program": {
                    "groups": 6,
                    "instructions": 6,
                    "alu": 5,
                    "tex": 0,
                    "memory": 0,
                },
                "executed": {"alu": 20480, "tex": 0, "memory": 0},
            },
        }
    ]
    artifact = output_dir / "varyings_shader_1_sample_000001.png"
    png_width, png_height, pixels = decode_rgba8_png(artifact)
    assert (png_width, png_height) == (64, 64)
    assert pixels == varyings_shader_one_golden_pixels()


def verify_varyings_shader_two(executable: Path, output_dir: Path) -> None:
    completed = invoke_case(
        executable,
        "varyings_shader_2",
        output_dir,
        frames=1,
        cache_bypass="off",
        width=64,
        height=64,
    )
    assert completed.returncode == 0, (
        "varyings_shader_2 model run failed:\n" + process_details(completed)
    )
    messages = json_messages(completed)
    hello = [message for message in messages if message.get("type") == "hello"]
    counters = [
        message for message in messages if message.get("type") == "counter"
    ]
    done = [message for message in messages if message.get("type") == "done"]
    assert len(hello) == len(counters) == len(done) == 1
    assert hello[0].get("mode") == "systemc-functional-varyings-shader"
    assert hello[0].get("functional_scope") == "varyings_shader_2-pco-iss-v1"
    assert (
        hello[0].get("pco_subset")
        == "fmul-fitrp-wdf-fadd-mbyp-uvsw-varying"
    )
    assert hello[0].get("shader_binary") == "mesa-pco-public-encoding"
    assert hello[0].get("cache_bypass") is False
    assert hello[0].get("face_cull_state") == {
        "enabled": False,
        "mode": "back",
        "front_face": "counter-clockwise",
    }
    assert done[0].get("pool_leaks") == 0
    assert done[0].get("pool_bytes_in_flight") == 0

    message = counters[0]
    assert message.get("provenance") == "modeled"
    assert message.get("vertex_pco_binary") == {
        "fingerprint": "fnv1a64:ffd408e5a8ae5f7c",
        "bytes": 120,
    }
    assert message.get("vertex_pco_opcodes") == {
        "fadd": 0,
        "fmul": 2,
        "mbyp": 6,
        "uvsw_write": 3,
        "uvsw_write_emit_endtask": 0,
        "uvsw_emit_endtask": 1,
    }
    assert message.get("fragment_pco_binary") == {
        "fingerprint": "fnv1a64:9c2a8c68ef09d5d1",
        "bytes": 104,
    }
    assert message.get("fragment_pco_opcodes") == {
        "fitrp": 2,
        "wdf": 2,
        "fadd": 4,
        "mbyp": 4,
    }
    values = message.get("counters")
    assert isinstance(values, dict)
    expected = {
        "ia_vertices": 96,
        "ia_primitives": 32,
        "vs_invocations": 25,
        "gs_invocations": 0,
        "gs_primitives": 0,
        "c_invocations": 32,
        "c_primitives": 32,
        "ps_invocations": 4096,
        "hs_invocations": 0,
        "ds_invocations": 0,
        "cs_invocations": 0,
        "ts_invocations": 0,
        "ms_invocations": 0,
        "ms_primitives": 0,
        "drawlists": 1,
        "setup_triangles": 32,
        "texel_fetches": 0,
        "vertex_attribute_fetches": 25,
        "vertex_attribute_bytes": 200,
        "pco_instructions": 24,
        "vs_alu_instructions": 200,
        "vs_tex_instructions": 0,
        "vs_memory_instructions": 325,
        "fs_alu_instructions": 40960,
        "fs_tex_instructions": 0,
        "fs_memory_instructions": 0,
        "fifo_stall_events": 0,
        "pool_bytes_in_flight": 0,
        "virtual_gpu_cycles": 1381,
        "tiler_cycles": 56,
        "renderer_cycles": 1300,
        "usc_groups": 1159,
        "vdm_cycles": 9,
        "vertex_fetch_cycles": 5,
        "pco_decode_cycles": 12,
        "usc_slot_cycles": 584,
        "usc_cluster_cycles": 298,
        "clip_cull_cycles": 7,
        "tiler_bin_cycles": 11,
        "parameter_buffer_cycles": 6,
        "parameter_coefficient_sets": 288,
        "parameter_write_bytes": 4608,
        "pds_coefficient_tasks": 1152,
        "pds_douti_issues": 2304,
        "usc_coefficient_load_bytes": 165888,
        "tile_scheduler_cycles": 9,
        "isp_cycles": 74,
        "fragment_frontend_cycles": 37,
        "texture_cycles": 2,
        "pbe_cycles": 44,
        "texture_requests": 0,
        "tiles_binned": 4,
        "tiles_scheduled": 4,
        "covered_pixels": 4096,
        "fragment_candidates": 4096,
        "hsr_rejected_fragments": 0,
        "depth_tested_fragments": 0,
        "depth_rejected_fragments": 0,
        "depth_written_fragments": 0,
        "pbe_color_reads": 0,
        "pbe_blended_fragments": 0,
        "pbe_fragment_writes": 4096,
        "pbe_pixels_written": 4096,
    }
    for field, wanted in expected.items():
        if field in UNIFIED_MEMORY_EXACT_SKIP:
            continue
        assert values.get(field) == wanted, (
            f"varyings_shader_2: {field}={values.get(field)!r}, expected {wanted}"
        )
    assert int(values.get("pool_high_water_bytes", 0)) > 0
    assert values["parameter_coefficient_sets"] == 32 * 9
    assert values["parameter_write_bytes"] == (
        values["parameter_coefficient_sets"] * 16
    )
    assert values["pds_coefficient_tasks"] == 1152
    assert values["pds_douti_issues"] == values["pds_coefficient_tasks"] * 2
    assert values["usc_coefficient_load_bytes"] == (
        values["pds_coefficient_tasks"] * 36 * 4
    )
    assert values["usc_groups"] == values["pds_coefficient_tasks"] + 7
    assert values["fragment_candidates"] == (
        values["ps_invocations"] + values["hsr_rejected_fragments"]
    )
    assert values["virtual_gpu_cycles"] == (
        values["tiler_cycles"] + values["renderer_cycles"] + 25
    )
    verify_memory_path(
        hello[0],
        message,
        cache_bypass=False,
        framebuffer_bytes=64 * 64 * 4,
    )
    assert message.get("drawlist_stats") == [
        {
            "drawlist": 0,
            "draw_id": 0,
            "vs": {
                "invocations": 25,
                "program": {
                    "groups": 12,
                    "instructions": 12,
                    "alu": 8,
                    "tex": 0,
                    "memory": 4,
                },
                "executed": {"alu": 200, "tex": 0, "memory": 325},
            },
            "fs": {
                "invocations": 4096,
                "program": {
                    "groups": 12,
                    "instructions": 12,
                    "alu": 10,
                    "tex": 0,
                    "memory": 0,
                },
                "executed": {"alu": 40960, "tex": 0, "memory": 0},
            },
        }
    ]
    artifact = output_dir / "varyings_shader_2_sample_000001.png"
    png_width, png_height, pixels = decode_rgba8_png(artifact)
    assert (png_width, png_height) == (64, 64)
    assert pixels == varyings_shader_two_golden_pixels()


def verify_varyings_shader_four(executable: Path, output_dir: Path) -> None:
    case_name = "varyings_shader_4"
    completed = invoke_case(
        executable,
        case_name,
        output_dir,
        frames=1,
        cache_bypass="off",
        width=64,
        height=64,
    )
    assert completed.returncode == 0, (
        f"{case_name} model run failed:\n" + process_details(completed)
    )
    messages = json_messages(completed)
    hello = [message for message in messages if message.get("type") == "hello"]
    counters = [
        message for message in messages if message.get("type") == "counter"
    ]
    done = [message for message in messages if message.get("type") == "done"]
    assert len(hello) == len(counters) == len(done) == 1
    assert hello[0].get("mode") == "systemc-functional-varyings-shader"
    assert hello[0].get("functional_scope") == f"{case_name}-pco-iss-v1"
    assert (
        hello[0].get("pco_subset")
        == "fmul-fitrp-wdf-fadd-mbyp-uvsw-varying"
    )
    assert hello[0].get("shader_binary") == "mesa-pco-public-encoding"
    assert hello[0].get("cache_bypass") is False
    assert hello[0].get("framebuffer_source") == "dram-readback"
    assert hello[0].get("dram_fixed_latency_cycles") == 1
    assert hello[0].get("tile_width") == 32
    assert hello[0].get("tile_height") == 32
    assert hello[0].get("face_cull_state") == {
        "enabled": False,
        "mode": "back",
        "front_face": "counter-clockwise",
    }
    assert done[0].get("pool_leaks") == 0
    assert done[0].get("pool_bytes_in_flight") == 0

    message = counters[0]
    assert message.get("provenance") == "modeled"
    assert message.get("vertex_pco_binary") == {
        "fingerprint": "fnv1a64:a654c923dfea45ef",
        "bytes": 136,
    }
    assert message.get("vertex_pco_opcodes") == {
        "fadd": 0,
        "fmul": 2,
        "mbyp": 6,
        "uvsw_write": 5,
        "uvsw_write_emit_endtask": 0,
        "uvsw_emit_endtask": 1,
    }
    assert message.get("fragment_pco_binary") == {
        "fingerprint": "fnv1a64:1dbf7c0552b6b385",
        "bytes": 216,
    }
    assert message.get("fragment_pco_opcodes") == {
        "fitrp": 4,
        "wdf": 4,
        "fadd": 12,
        "mbyp": 4,
    }

    values = message.get("counters")
    assert isinstance(values, dict)
    expected = {
        "ia_vertices": 96,
        "ia_primitives": 32,
        "vs_invocations": 25,
        "gs_invocations": 0,
        "gs_primitives": 0,
        "c_invocations": 32,
        "c_primitives": 32,
        "ps_invocations": 4096,
        "hs_invocations": 0,
        "ds_invocations": 0,
        "cs_invocations": 0,
        "ts_invocations": 0,
        "ms_invocations": 0,
        "ms_primitives": 0,
        "drawlists": 1,
        "setup_triangles": 32,
        "texel_fetches": 0,
        "virtual_gpu_cycles": 1385,
        "tiler_cycles": 57,
        "renderer_cycles": 1303,
        "usc_groups": 1159,
        "texture_requests": 0,
        "fifo_stall_events": 0,
        "pool_bytes_in_flight": 0,
        "pool_high_water_bytes": 1037272,
        "vdm_cycles": 9,
        "vertex_fetch_cycles": 5,
        "vertex_attribute_fetches": 25,
        "vertex_attribute_bytes": 200,
        "pco_decode_cycles": 16,
        "pco_instructions": 38,
        "vs_alu_instructions": 200,
        "vs_tex_instructions": 0,
        "vs_memory_instructions": 525,
        "fs_alu_instructions": 81920,
        "fs_tex_instructions": 0,
        "fs_memory_instructions": 0,
        "usc_slot_cycles": 584,
        "usc_cluster_cycles": 298,
        "clip_cull_cycles": 7,
        "tiler_bin_cycles": 11,
        "parameter_buffer_cycles": 6,
        "parameter_coefficient_sets": 544,
        "parameter_write_bytes": 8704,
        "pds_coefficient_tasks": 1152,
        "pds_douti_issues": 2304,
        "usc_coefficient_load_bytes": 313344,
        "tile_scheduler_cycles": 9,
        "isp_cycles": 74,
        "fragment_frontend_cycles": 37,
        "texture_cycles": 2,
        "pbe_cycles": 44,
        "pixel_data_master_transactions": 1,
        "pixel_data_master_bytes": 16384,
        "pixel_data_master_cycles": 1,
        "slc_line_accesses": 128,
        "slc_read_accesses": 0,
        "slc_write_accesses": 128,
        "slc_hits": 0,
        "slc_misses": 128,
        "slc_evictions": 0,
        "slc_writebacks": 128,
        "slc_bypassed": 0,
        "slc_cycles": 128,
        "dram_read_transactions": 1,
        "dram_write_transactions": 128,
        "dram_read_bytes": 16384,
        "dram_write_bytes": 16384,
        "dram_cycles": 129,
        "framebuffer_dram_readback_bytes": 16384,
        "tiles_binned": 4,
        "tiles_scheduled": 4,
        "covered_pixels": 4096,
        "fragment_candidates": 4096,
        "hsr_rejected_fragments": 0,
        "depth_tested_fragments": 0,
        "depth_rejected_fragments": 0,
        "depth_written_fragments": 0,
        "pbe_color_reads": 0,
        "pbe_blended_fragments": 0,
        "pbe_fragment_writes": 4096,
        "pbe_pixels_written": 4096,
        "functional_frame": 1,
    }
    for field, wanted in expected.items():
        if field in UNIFIED_MEMORY_EXACT_SKIP:
            continue
        assert values.get(field) == wanted, (
            f"{case_name}: {field}={values.get(field)!r}, expected {wanted}"
        )

    assert values["parameter_coefficient_sets"] == values["c_primitives"] * 17
    assert values["parameter_write_bytes"] == (
        values["parameter_coefficient_sets"] * 16
    )
    assert values["pds_douti_issues"] == values["pds_coefficient_tasks"] * 2
    assert values["usc_coefficient_load_bytes"] == (
        values["pds_coefficient_tasks"] * 68 * 4
    )
    assert values["usc_groups"] == values["pds_coefficient_tasks"] + 7
    assert values["fragment_candidates"] == (
        values["ps_invocations"] + values["hsr_rejected_fragments"]
    )
    assert values["pbe_fragment_writes"] == values["ps_invocations"]
    assert values["pixel_data_master_bytes"] == values["pbe_pixels_written"] * 4
    assert values["dram_cycles"] == (
        values["dram_read_transactions"] + values["dram_write_transactions"]
    )
    assert values["virtual_gpu_cycles"] == (
        values["tiler_cycles"] + values["renderer_cycles"] + 25
    )
    verify_memory_path(
        hello[0],
        message,
        cache_bypass=False,
        framebuffer_bytes=64 * 64 * 4,
    )
    assert message.get("drawlist_stats") == [
        {
            "drawlist": 0,
            "draw_id": 0,
            "vs": {
                "invocations": 25,
                "program": {
                    "groups": 14,
                    "instructions": 14,
                    "alu": 8,
                    "tex": 0,
                    "memory": 6,
                },
                "executed": {"alu": 200, "tex": 0, "memory": 525},
            },
            "fs": {
                "invocations": 4096,
                "program": {
                    "groups": 24,
                    "instructions": 24,
                    "alu": 20,
                    "tex": 0,
                    "memory": 0,
                },
                "executed": {"alu": 81920, "tex": 0, "memory": 0},
            },
        }
    ]
    artifact = output_dir / f"{case_name}_sample_000001.png"
    png_width, png_height, pixels = decode_rgba8_png(artifact)
    assert (png_width, png_height) == (64, 64)
    assert pixels == varyings_shader_four_golden_pixels()


def verify_varyings_shader_eight(executable: Path, output_dir: Path) -> None:
    case_name = "varyings_shader_8"
    completed = invoke_case(
        executable,
        case_name,
        output_dir,
        frames=1,
        cache_bypass="off",
        width=64,
        height=64,
    )
    assert completed.returncode == 0, (
        f"{case_name} model run failed:\n" + process_details(completed)
    )
    messages = json_messages(completed)
    hello = [message for message in messages if message.get("type") == "hello"]
    counters = [
        message for message in messages if message.get("type") == "counter"
    ]
    done = [message for message in messages if message.get("type") == "done"]
    assert len(hello) == len(counters) == len(done) == 1
    assert hello[0].get("mode") == "systemc-functional-varyings-shader"
    assert hello[0].get("functional_scope") == f"{case_name}-pco-iss-v1"
    assert (
        hello[0].get("pco_subset")
        == "fmul-fitrp-wdf-fadd-mbyp-uvsw-varying"
    )
    assert hello[0].get("shader_binary") == "mesa-pco-public-encoding"
    assert hello[0].get("cache_bypass") is False
    assert hello[0].get("framebuffer_source") == "dram-readback"
    assert hello[0].get("dram_fixed_latency_cycles") == 1
    assert hello[0].get("tile_width") == 32
    assert hello[0].get("tile_height") == 32
    assert hello[0].get("face_cull_state") == {
        "enabled": False,
        "mode": "back",
        "front_face": "counter-clockwise",
    }
    assert done[0].get("pool_leaks") == 0
    assert done[0].get("pool_bytes_in_flight") == 0

    message = counters[0]
    assert message.get("provenance") == "modeled"
    assert message.get("vertex_pco_binary") == {
        "fingerprint": "fnv1a64:3ea4e650a43484ce",
        "bytes": 176,
    }
    assert message.get("vertex_pco_opcodes") == {
        "fadd": 0,
        "fmul": 2,
        "mbyp": 6,
        "uvsw_write": 9,
        "uvsw_write_emit_endtask": 0,
        "uvsw_emit_endtask": 1,
    }
    assert message.get("fragment_pco_binary") == {
        "fingerprint": "fnv1a64:b1f3b2aa7d58d59d",
        "bytes": 440,
    }
    assert message.get("fragment_pco_opcodes") == {
        "fitrp": 8,
        "wdf": 8,
        "fadd": 28,
        "mbyp": 4,
    }

    values = message.get("counters")
    assert isinstance(values, dict)
    expected = {
        "ia_vertices": 96,
        "ia_primitives": 32,
        "vs_invocations": 25,
        "gs_invocations": 0,
        "gs_primitives": 0,
        "c_invocations": 32,
        "c_primitives": 32,
        "ps_invocations": 4096,
        "hs_invocations": 0,
        "ds_invocations": 0,
        "cs_invocations": 0,
        "ts_invocations": 0,
        "ms_invocations": 0,
        "ms_primitives": 0,
        "drawlists": 1,
        "setup_triangles": 32,
        "texel_fetches": 0,
        "virtual_gpu_cycles": 1392,
        "tiler_cycles": 58,
        "renderer_cycles": 1309,
        "usc_groups": 1159,
        "texture_requests": 0,
        "fifo_stall_events": 0,
        "pool_bytes_in_flight": 0,
        "pool_high_water_bytes": 1347944,
        "vdm_cycles": 9,
        "vertex_fetch_cycles": 5,
        "vertex_attribute_fetches": 25,
        "vertex_attribute_bytes": 200,
        "pco_decode_cycles": 23,
        "pco_instructions": 66,
        "vs_alu_instructions": 200,
        "vs_tex_instructions": 0,
        "vs_memory_instructions": 925,
        "fs_alu_instructions": 163840,
        "fs_tex_instructions": 0,
        "fs_memory_instructions": 0,
        "usc_slot_cycles": 584,
        "usc_cluster_cycles": 298,
        "clip_cull_cycles": 7,
        "tiler_bin_cycles": 11,
        "parameter_buffer_cycles": 6,
        "parameter_coefficient_sets": 1056,
        "parameter_write_bytes": 16896,
        "pds_coefficient_tasks": 1152,
        "pds_douti_issues": 2304,
        "usc_coefficient_load_bytes": 608256,
        "tile_scheduler_cycles": 9,
        "isp_cycles": 74,
        "fragment_frontend_cycles": 37,
        "texture_cycles": 2,
        "pbe_cycles": 44,
        "pixel_data_master_transactions": 1,
        "pixel_data_master_bytes": 16384,
        "pixel_data_master_cycles": 1,
        "slc_line_accesses": 128,
        "slc_read_accesses": 0,
        "slc_write_accesses": 128,
        "slc_hits": 0,
        "slc_misses": 128,
        "slc_evictions": 0,
        "slc_writebacks": 128,
        "slc_bypassed": 0,
        "slc_cycles": 128,
        "dram_read_transactions": 1,
        "dram_write_transactions": 128,
        "dram_read_bytes": 16384,
        "dram_write_bytes": 16384,
        "dram_cycles": 129,
        "framebuffer_dram_readback_bytes": 16384,
        "tiles_binned": 4,
        "tiles_scheduled": 4,
        "covered_pixels": 4096,
        "fragment_candidates": 4096,
        "hsr_rejected_fragments": 0,
        "depth_tested_fragments": 0,
        "depth_rejected_fragments": 0,
        "depth_written_fragments": 0,
        "pbe_color_reads": 0,
        "pbe_blended_fragments": 0,
        "pbe_fragment_writes": 4096,
        "pbe_pixels_written": 4096,
        "functional_frame": 1,
    }
    for field, wanted in expected.items():
        if field in UNIFIED_MEMORY_EXACT_SKIP:
            continue
        assert values.get(field) == wanted, (
            f"{case_name}: {field}={values.get(field)!r}, expected {wanted}"
        )

    assert values["parameter_coefficient_sets"] == values["c_primitives"] * 33
    assert values["parameter_write_bytes"] == (
        values["parameter_coefficient_sets"] * 16
    )
    assert values["pds_douti_issues"] == values["pds_coefficient_tasks"] * 2
    assert values["usc_coefficient_load_bytes"] == (
        values["pds_coefficient_tasks"] * 132 * 4
    )
    assert values["usc_groups"] == values["pds_coefficient_tasks"] + 7
    assert values["fragment_candidates"] == (
        values["ps_invocations"] + values["hsr_rejected_fragments"]
    )
    assert values["pbe_fragment_writes"] == values["ps_invocations"]
    assert values["pixel_data_master_bytes"] == values["pbe_pixels_written"] * 4
    assert values["dram_cycles"] == (
        values["dram_read_transactions"] + values["dram_write_transactions"]
    )
    assert values["virtual_gpu_cycles"] == (
        values["tiler_cycles"] + values["renderer_cycles"] + 25
    )
    verify_memory_path(
        hello[0],
        message,
        cache_bypass=False,
        framebuffer_bytes=64 * 64 * 4,
    )
    assert message.get("drawlist_stats") == [
        {
            "drawlist": 0,
            "draw_id": 0,
            "vs": {
                "invocations": 25,
                "program": {
                    "groups": 18,
                    "instructions": 18,
                    "alu": 8,
                    "tex": 0,
                    "memory": 10,
                },
                "executed": {"alu": 200, "tex": 0, "memory": 925},
            },
            "fs": {
                "invocations": 4096,
                "program": {
                    "groups": 48,
                    "instructions": 48,
                    "alu": 40,
                    "tex": 0,
                    "memory": 0,
                },
                "executed": {"alu": 163840, "tex": 0, "memory": 0},
            },
        }
    ]
    artifact = output_dir / f"{case_name}_sample_000001.png"
    png_width, png_height, pixels = decode_rgba8_png(artifact)
    assert (png_width, png_height) == (64, 64)
    assert pixels == varyings_shader_eight_golden_pixels()


def verify_fill_tex_nearest(executable: Path, output_dir: Path) -> None:
    case_name = "fill_tex_nearest"
    completed = invoke_case(
        executable,
        case_name,
        output_dir,
        frames=1,
        cache_bypass="off",
        width=64,
        height=64,
    )
    assert completed.returncode == 0, (
        f"{case_name} model run failed:\n" + process_details(completed)
    )

    messages = json_messages(completed)
    hello = [message for message in messages if message.get("type") == "hello"]
    counters = [
        message for message in messages if message.get("type") == "counter"
    ]
    done = [message for message in messages if message.get("type") == "done"]
    assert len(hello) == len(counters) == len(done) == 1

    hello_message = hello[0]
    expected_hello = {
        "backend": "pvrgpu",
        "mode": "systemc-functional-fill-texture-nearest",
        "functional_scope": f"{case_name}-pco-iss-v1",
        "command_source": "builtin-glbench-fixture",
        "shader_binary": "mesa-pco-public-encoding",
        "pco_subset": "fmul-fitrp-wdf-smp-mbyp-uvsw-texture",
        "reference_uarch": "pvrgpu-ref-v1",
        "uarch_provenance": "assumed",
        "timing_provenance": "uncalibrated",
        "cache_bypass": False,
        "cache_policy": "set-associative-write-back-write-allocate-true-lru",
        "framebuffer_source": "dram-readback",
        "dram_fixed_latency_cycles": 1,
        "tile_width": 32,
        "tile_height": 32,
    }
    for field, expected in expected_hello.items():
        assert hello_message.get(field) == expected, (
            f"{case_name}: hello {field}={hello_message.get(field)!r}, "
            f"expected {expected!r}"
        )
    assert hello_message.get("tcu_cache") == {
        "capacity_bytes": 24 * 1024,
        "line_bytes": 64,
        "ways": 4,
        "banks": 4,
    }
    assert hello_message.get("slc_cache") == {
        "capacity_bytes": 2 * 1024 * 1024,
        "line_bytes": 128,
        "ways": 8,
        "banks": 8,
    }
    done_message = done[0]
    assert done_message.get("pool_leaks") == 0
    assert done_message.get("pool_bytes_in_flight") == 0
    assert done_message.get("pool_allocations") == done_message.get(
        "pool_releases"
    )

    message = counters[0]
    expected_provenance = {
        "backend": "pvrgpu",
        "source": "pvrgpu-systemc",
        "provenance": "modeled",
        "functional_scope": f"{case_name}-pco-iss-v1",
        "command_source": "builtin-glbench-fixture",
        "timing_provenance": "uncalibrated",
        "cache_bypass": False,
        "framebuffer_source": "dram-readback",
    }
    for field, expected in expected_provenance.items():
        if field == "virtual_time_ns":
            continue
        assert message.get(field) == expected, (
            f"{case_name}: counter {field}={message.get(field)!r}, "
            f"expected {expected!r}"
        )

    assert message.get("vertex_pco_binary") == {
        "fingerprint": "fnv1a64:36c31424e4119557",
        "bytes": 80,
    }
    assert message.get("vertex_pco_opcodes") == {
        "fadd": 0,
        "fmul": 2,
        "mbyp": 3,
        "uvsw_write": 2,
        "uvsw_write_emit_endtask": 0,
        "uvsw_emit_endtask": 1,
    }
    assert message.get("fragment_pco_binary") == {
        "fingerprint": "fnv1a64:0693891931816150",
        "bytes": 184,
    }
    assert message.get("fragment_pco_opcodes") == {
        "fitrp": 1,
        "wdf": 2,
        "smp": 1,
        "mbyp": 18,
    }

    values = message.get("counters")
    assert isinstance(values, dict), f"{case_name}: counters is not an object"
    expected_standard = {
        "ia_vertices": 4,
        "ia_primitives": 2,
        "vs_invocations": 4,
        "gs_invocations": 0,
        "gs_primitives": 0,
        "c_invocations": 2,
        "c_primitives": 2,
        "ps_invocations": 4096,
        "hs_invocations": 0,
        "ds_invocations": 0,
        "cs_invocations": 0,
        "ts_invocations": 0,
        "ms_invocations": 0,
        "ms_primitives": 0,
        "drawlists": 1,
        "setup_triangles": 2,
        "texel_fetches": 4352,
    }
    for field, expected in expected_standard.items():
        assert values.get(field) == expected, (
            f"{case_name}: {field}={values.get(field)!r}, expected {expected}"
        )

    expected_modeled = {
        "virtual_gpu_cycles": 17459,
        "tiler_cycles": 51,
        "renderer_cycles": 17383,
        "usc_groups": 1089,
        "texture_requests": 4352,
        "fifo_stall_events": 0,
        "pool_bytes_in_flight": 0,
        # Includes quad_id/quad_lane in the TextureSampleRequest ABI.
        "pool_high_water_bytes": 3246354,
        "vdm_cycles": 9,
        "vertex_fetch_cycles": 5,
        "vertex_attribute_fetches": 8,
        "vertex_attribute_bytes": 64,
        "pco_decode_cycles": 14,
        "pco_instructions": 30,
        "vs_alu_instructions": 20,
        "vs_tex_instructions": 0,
        "vs_memory_instructions": 28,
        "fs_alu_instructions": 82688,
        "fs_tex_instructions": 4352,
        "fs_memory_instructions": 0,
        "usc_slot_cycles": 549,
        "usc_cluster_cycles": 281,
        "clip_cull_cycles": 7,
        "tiler_bin_cycles": 11,
        "parameter_buffer_cycles": 6,
        "parameter_coefficient_sets": 6,
        "parameter_write_bytes": 96,
        "pds_coefficient_tasks": 1088,
        "pds_douti_issues": 2176,
        "usc_coefficient_load_bytes": 52224,
        "tile_scheduler_cycles": 9,
        "isp_cycles": 74,
        "fragment_frontend_cycles": 39,
        "texture_cycles": 8704,
        "pbe_cycles": 44,
        "pixel_data_master_transactions": 1,
        "pixel_data_master_bytes": 16384,
        "pixel_data_master_cycles": 1,
        "tcu_line_accesses": 4352,
        "tcu_read_accesses": 4352,
        "tcu_hits": 2304,
        "tcu_misses": 2048,
        "tcu_evictions": 1664,
        "tcu_writebacks": 0,
        "tcu_bypassed": 0,
        "tcu_cycles": 4352,
        "slc_line_accesses": 2176,
        "slc_read_accesses": 2048,
        "slc_write_accesses": 128,
        "slc_hits": 1024,
        "slc_misses": 1152,
        "slc_evictions": 0,
        "slc_writebacks": 128,
        "slc_bypassed": 0,
        "slc_cycles": 2176,
        "dram_read_transactions": 1025,
        "dram_write_transactions": 128,
        "dram_read_bytes": 147456,
        "dram_write_bytes": 16384,
        "dram_cycles": 1153,
        "framebuffer_dram_readback_bytes": 16384,
        "tiles_binned": 4,
        "tiles_scheduled": 4,
        "covered_pixels": 4096,
        "fragment_candidates": 4096,
        "hsr_rejected_fragments": 0,
        "depth_tested_fragments": 0,
        "depth_rejected_fragments": 0,
        "depth_written_fragments": 0,
        "pbe_color_reads": 0,
        "pbe_blended_fragments": 0,
        "pbe_fragment_writes": 4096,
        "pbe_pixels_written": 4096,
        "functional_frame": 1,
    }
    for field, expected in expected_modeled.items():
        if field in UNIFIED_MEMORY_EXACT_SKIP:
            continue
        assert values.get(field) == expected, (
            f"{case_name}: {field}={values.get(field)!r}, expected {expected}"
        )

    drawlists = message.get("drawlist_stats")
    assert isinstance(drawlists, list) and len(drawlists) == 1
    drawlist = drawlists[0]
    assert isinstance(drawlist, dict)
    assert drawlist.get("drawlist") == 0 and drawlist.get("draw_id") == 0
    vertex = drawlist.get("vs")
    fragment = drawlist.get("fs")
    assert isinstance(vertex, dict) and isinstance(fragment, dict)
    assert vertex.get("invocations") == 4
    assert fragment.get("invocations") == 4096
    vertex_program = vertex.get("program")
    vertex_executed = vertex.get("executed")
    fragment_program = fragment.get("program")
    fragment_executed = fragment.get("executed")
    assert all(
        isinstance(item, dict)
        for item in (
            vertex_program,
            vertex_executed,
            fragment_program,
            fragment_executed,
        )
    )
    assert vertex_program == {
        "groups": 8,
        "instructions": 8,
        "alu": 5,
        "tex": 0,
        "memory": 3,
    }
    assert vertex_executed == {"alu": 20, "tex": 0, "memory": 28}
    assert fragment_program == {
        "groups": 22,
        "instructions": 22,
        "alu": 19,
        "tex": 1,
        "memory": 0,
    }
    assert fragment_executed == {"alu": 82688, "tex": 4352, "memory": 0}

    assert values["parameter_coefficient_sets"] == values["c_primitives"] * 3
    assert values["parameter_write_bytes"] == (
        values["parameter_coefficient_sets"] * 4 * 4
    )
    assert values["pds_douti_issues"] == values["pds_coefficient_tasks"] * 2
    assert values["usc_coefficient_load_bytes"] == (
        values["pds_coefficient_tasks"] * 12 * 4
    )
    assert values["usc_groups"] == values["pds_coefficient_tasks"] + 1
    assert values["pco_instructions"] == (
        vertex_program["instructions"] + fragment_program["instructions"]
    )
    assert values["vs_alu_instructions"] == vertex_executed["alu"]
    assert values["vs_tex_instructions"] == vertex_executed["tex"]
    assert values["vs_memory_instructions"] == vertex_executed["memory"]
    assert values["fs_alu_instructions"] == fragment_executed["alu"]
    assert values["fs_tex_instructions"] == fragment_executed["tex"]
    assert values["fs_memory_instructions"] == fragment_executed["memory"]
    assert values["texture_requests"] == values["texel_fetches"]
    assert values["fs_tex_instructions"] == values["texture_requests"]
    assert values["fs_alu_instructions"] == (
        values["texture_requests"] * fragment_program["alu"]
    )
    assert values["texture_cycles"] >= values["texture_requests"]
    assert values["fragment_candidates"] == (
        values["ps_invocations"] + values["hsr_rejected_fragments"]
    )
    assert values["pbe_fragment_writes"] == values["ps_invocations"]
    assert values["pixel_data_master_bytes"] == values["pbe_pixels_written"] * 4
    assert values["virtual_gpu_cycles"] == (
        values["tiler_cycles"] + values["renderer_cycles"] + 25
    )
    verify_memory_path(
        hello_message,
        message,
        cache_bypass=False,
        framebuffer_bytes=64 * 64 * 4,
    )

    artifact = output_dir / f"{case_name}_sample_000001.png"
    artifact_field = message.get("artifact_png")
    assert isinstance(artifact_field, str)
    assert Path(artifact_field).resolve() == artifact.resolve()
    png_width, png_height, pixels = decode_rgba8_png(artifact)
    assert (png_width, png_height) == (64, 64)
    assert pixels == fill_tex_nearest_golden_pixels()
    assert [path.name for path in output_dir.rglob("*.png")] == [artifact.name]
    assert not [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name.endswith(".part")
    ]


def verify_fill_tex_bilinear(executable: Path, output_dir: Path) -> None:
    case_name = "fill_tex_bilinear"
    completed = invoke_case(
        executable,
        case_name,
        output_dir,
        frames=1,
        cache_bypass="off",
        width=64,
        height=64,
    )
    assert completed.returncode == 0, (
        f"{case_name} model run failed:\n" + process_details(completed)
    )

    messages = json_messages(completed)
    hello = [message for message in messages if message.get("type") == "hello"]
    counters = [
        message for message in messages if message.get("type") == "counter"
    ]
    done = [message for message in messages if message.get("type") == "done"]
    assert len(hello) == len(counters) == len(done) == 1

    hello_message = hello[0]
    expected_hello = {
        "backend": "pvrgpu",
        "mode": "systemc-functional-fill-texture-bilinear",
        "functional_scope": f"{case_name}-pco-iss-v1",
        "command_source": "builtin-glbench-fixture",
        "shader_binary": "mesa-pco-public-encoding",
        "pco_subset": "fmul-fitrp-wdf-smp-mbyp-uvsw-texture",
        "workload": case_name,
        "reference_uarch": "pvrgpu-ref-v1",
        "uarch_provenance": "assumed",
        "timing_provenance": "uncalibrated",
        "cache_bypass": False,
        "cache_policy": "set-associative-write-back-write-allocate-true-lru",
        "framebuffer_source": "dram-readback",
        "dram_fixed_latency_cycles": 1,
        "tile_width": 32,
        "tile_height": 32,
    }
    for field, expected in expected_hello.items():
        assert hello_message.get(field) == expected, (
            f"{case_name}: hello {field}={hello_message.get(field)!r}, "
            f"expected {expected!r}"
        )
    assert hello_message.get("tcu_cache") == {
        "capacity_bytes": 24 * 1024,
        "line_bytes": 64,
        "ways": 4,
        "banks": 4,
    }
    assert hello_message.get("slc_cache") == {
        "capacity_bytes": 2 * 1024 * 1024,
        "line_bytes": 128,
        "ways": 8,
        "banks": 8,
    }

    done_message = done[0]
    assert done_message.get("pool_leaks") == 0
    assert done_message.get("pool_bytes_in_flight") == 0
    assert done_message.get("pool_allocations") == done_message.get(
        "pool_releases"
    )

    message = counters[0]
    expected_provenance = {
        "backend": "pvrgpu",
        "source": "pvrgpu-systemc",
        "provenance": "modeled",
        "functional_scope": f"{case_name}-pco-iss-v1",
        "command_source": "builtin-glbench-fixture",
        "timing_provenance": "uncalibrated",
        "cache_bypass": False,
        "framebuffer_source": "dram-readback",
        "marker": case_name,
    }
    for field, expected in expected_provenance.items():
        if field == "virtual_time_ns":
            continue
        assert message.get(field) == expected, (
            f"{case_name}: counter {field}={message.get(field)!r}, "
            f"expected {expected!r}"
        )

    assert message.get("vertex_pco_binary") == {
        "fingerprint": "fnv1a64:36c31424e4119557",
        "bytes": 80,
    }
    assert message.get("vertex_pco_opcodes") == {
        "fadd": 0,
        "fmul": 2,
        "mbyp": 3,
        "uvsw_write": 2,
        "uvsw_write_emit_endtask": 0,
        "uvsw_emit_endtask": 1,
    }
    assert message.get("fragment_pco_binary") == {
        "fingerprint": "fnv1a64:0693891931816150",
        "bytes": 184,
    }
    assert message.get("fragment_pco_opcodes") == {
        "fitrp": 1,
        "wdf": 2,
        "smp": 1,
        "mbyp": 18,
    }

    values = message.get("counters")
    assert isinstance(values, dict), f"{case_name}: counters is not an object"
    expected_standard = {
        "ia_vertices": 4,
        "ia_primitives": 2,
        "vs_invocations": 4,
        "gs_invocations": 0,
        "gs_primitives": 0,
        "c_invocations": 2,
        "c_primitives": 2,
        "ps_invocations": 4096,
        "hs_invocations": 0,
        "ds_invocations": 0,
        "cs_invocations": 0,
        "ts_invocations": 0,
        "ms_invocations": 0,
        "ms_primitives": 0,
        "drawlists": 1,
        "setup_triangles": 2,
        "texel_fetches": 17408,
    }
    for field, expected in expected_standard.items():
        assert values.get(field) == expected, (
            f"{case_name}: {field}={values.get(field)!r}, expected {expected}"
        )

    expected_modeled = {
        "virtual_gpu_cycles": 33587,
        "tiler_cycles": 51,
        "renderer_cycles": 33511,
        "usc_groups": 1089,
        "texture_requests": 4352,
        "fifo_stall_events": 0,
        "pool_bytes_in_flight": 0,
        # Includes quad_id/quad_lane in the TextureSampleRequest ABI.
        "pool_high_water_bytes": 3246354,
        "vdm_cycles": 9,
        "vertex_fetch_cycles": 5,
        "vertex_attribute_fetches": 8,
        "vertex_attribute_bytes": 64,
        "pco_decode_cycles": 14,
        "pco_instructions": 30,
        "vs_alu_instructions": 20,
        "vs_tex_instructions": 0,
        "vs_memory_instructions": 28,
        "fs_alu_instructions": 82688,
        "fs_tex_instructions": 4352,
        "fs_memory_instructions": 0,
        "usc_slot_cycles": 549,
        "usc_cluster_cycles": 281,
        "clip_cull_cycles": 7,
        "tiler_bin_cycles": 11,
        "parameter_buffer_cycles": 6,
        "parameter_coefficient_sets": 6,
        "parameter_write_bytes": 96,
        "pds_coefficient_tasks": 1088,
        "pds_douti_issues": 2176,
        "usc_coefficient_load_bytes": 52224,
        "tile_scheduler_cycles": 9,
        "isp_cycles": 74,
        "fragment_frontend_cycles": 39,
        "texture_cycles": 8704,
        "pbe_cycles": 44,
        "pixel_data_master_transactions": 1,
        "pixel_data_master_bytes": 16384,
        "pixel_data_master_cycles": 1,
        "tcu_line_accesses": 17408,
        "tcu_read_accesses": 17408,
        "tcu_hits": 13312,
        "tcu_misses": 4096,
        "tcu_evictions": 3712,
        "tcu_writebacks": 0,
        "tcu_bypassed": 0,
        "tcu_cycles": 17408,
        "slc_line_accesses": 4224,
        "slc_read_accesses": 4096,
        "slc_write_accesses": 128,
        "slc_hits": 2048,
        "slc_misses": 2176,
        "slc_evictions": 0,
        "slc_writebacks": 128,
        "slc_bypassed": 0,
        "slc_cycles": 4224,
        "dram_read_transactions": 2049,
        "dram_write_transactions": 128,
        "dram_read_bytes": 278528,
        "dram_write_bytes": 16384,
        "dram_cycles": 2177,
        "framebuffer_dram_readback_bytes": 16384,
        "tiles_binned": 4,
        "tiles_scheduled": 4,
        "covered_pixels": 4096,
        "fragment_candidates": 4096,
        "hsr_rejected_fragments": 0,
        "depth_tested_fragments": 0,
        "depth_rejected_fragments": 0,
        "depth_written_fragments": 0,
        "pbe_color_reads": 0,
        "pbe_blended_fragments": 0,
        "pbe_fragment_writes": 4096,
        "pbe_pixels_written": 4096,
        "functional_frame": 1,
    }
    for field, expected in expected_modeled.items():
        if field in UNIFIED_MEMORY_EXACT_SKIP:
            continue
        assert values.get(field) == expected, (
            f"{case_name}: {field}={values.get(field)!r}, expected {expected}"
        )

    assert message.get("drawlist_stats") == [
        {
            "drawlist": 0,
            "draw_id": 0,
            "vs": {
                "invocations": 4,
                "program": {
                    "groups": 8,
                    "instructions": 8,
                    "alu": 5,
                    "tex": 0,
                    "memory": 3,
                },
                "executed": {"alu": 20, "tex": 0, "memory": 28},
            },
            "fs": {
                "invocations": 4096,
                "program": {
                    "groups": 22,
                    "instructions": 22,
                    "alu": 19,
                    "tex": 1,
                    "memory": 0,
                },
                "executed": {"alu": 82688, "tex": 4352, "memory": 0},
            },
        }
    ]

    assert values["parameter_coefficient_sets"] == values["c_primitives"] * 3
    assert values["parameter_write_bytes"] == (
        values["parameter_coefficient_sets"] * 4 * 4
    )
    assert values["pds_douti_issues"] == values["pds_coefficient_tasks"] * 2
    assert values["usc_coefficient_load_bytes"] == (
        values["pds_coefficient_tasks"] * 12 * 4
    )
    assert values["usc_groups"] == values["pds_coefficient_tasks"] + 1
    assert values["texture_requests"] == values["fs_tex_instructions"]
    assert values["texel_fetches"] == values["texture_requests"] * 4
    assert values["fs_alu_instructions"] == values["texture_requests"] * 19
    assert values["texture_cycles"] >= values["texture_requests"]
    assert values["fragment_candidates"] == (
        values["ps_invocations"] + values["hsr_rejected_fragments"]
    )
    assert values["pbe_fragment_writes"] == values["ps_invocations"]
    assert values["pixel_data_master_bytes"] == values["pbe_pixels_written"] * 4
    assert values["virtual_gpu_cycles"] == (
        values["tiler_cycles"] + values["renderer_cycles"] + 25
    )
    verify_memory_path(
        hello_message,
        message,
        cache_bypass=False,
        framebuffer_bytes=64 * 64 * 4,
    )

    artifact = output_dir / f"{case_name}_sample_000001.png"
    artifact_field = message.get("artifact_png")
    assert isinstance(artifact_field, str)
    assert Path(artifact_field).resolve() == artifact.resolve()
    png_width, png_height, pixels = decode_rgba8_png(artifact)
    assert (png_width, png_height) == (64, 64)
    assert pixels == fill_tex_bilinear_golden_pixels()
    assert [path.name for path in output_dir.rglob("*.png")] == [artifact.name]
    assert not [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name.endswith(".part")
    ]


def verify_fill_tex_trilinear_linear_01(
    executable: Path, output_dir: Path
) -> None:
    case_name = "fill_tex_trilinear_linear_01"
    completed = invoke_case(
        executable,
        case_name,
        output_dir,
        frames=1,
        cache_bypass="off",
        width=64,
        height=64,
    )
    assert completed.returncode == 0, (
        f"{case_name} model run failed:\n" + process_details(completed)
    )

    messages = json_messages(completed)
    hello = [message for message in messages if message.get("type") == "hello"]
    counters = [
        message for message in messages if message.get("type") == "counter"
    ]
    done = [message for message in messages if message.get("type") == "done"]
    assert len(hello) == len(counters) == len(done) == 1

    hello_message = hello[0]
    expected_hello = {
        "backend": "pvrgpu",
        "mode": "systemc-functional-fill-texture-trilinear-linear-01",
        "functional_scope": f"{case_name}-pco-iss-v1",
        "command_source": "builtin-glbench-fixture",
        "shader_binary": "mesa-pco-public-encoding",
        "pco_subset": "fmul-fitrp-wdf-smp-mbyp-uvsw-texture",
        "reference_uarch": "pvrgpu-ref-v1",
        "uarch_provenance": "assumed",
        "timing_provenance": "uncalibrated",
        "cache_bypass": False,
        "cache_policy": "set-associative-write-back-write-allocate-true-lru",
        "framebuffer_source": "dram-readback",
        "dram_fixed_latency_cycles": 1,
        "tile_width": 32,
        "tile_height": 32,
        "workload": case_name,
    }
    for field, expected in expected_hello.items():
        assert hello_message.get(field) == expected, (
            f"{case_name}: hello {field}={hello_message.get(field)!r}, "
            f"expected {expected!r}"
        )
    assert hello_message.get("tcu_cache") == {
        "capacity_bytes": 24 * 1024,
        "line_bytes": 64,
        "ways": 4,
        "banks": 4,
    }
    assert hello_message.get("slc_cache") == {
        "capacity_bytes": 2 * 1024 * 1024,
        "line_bytes": 128,
        "ways": 8,
        "banks": 8,
    }

    done_message = done[0]
    assert done_message.get("pool_leaks") == 0
    assert done_message.get("pool_bytes_in_flight") == 0
    assert done_message.get("pool_allocations") == done_message.get(
        "pool_releases"
    )

    message = counters[0]
    expected_provenance = {
        "backend": "pvrgpu",
        "source": "pvrgpu-systemc",
        "provenance": "modeled",
        "functional_scope": f"{case_name}-pco-iss-v1",
        "command_source": "builtin-glbench-fixture",
        "timing_provenance": "uncalibrated",
        "cache_bypass": False,
        "framebuffer_source": "dram-readback",
        "marker": case_name,
    }
    for field, expected in expected_provenance.items():
        if field == "virtual_time_ns":
            continue
        assert message.get(field) == expected, (
            f"{case_name}: counter {field}={message.get(field)!r}, "
            f"expected {expected!r}"
        )

    assert message.get("vertex_pco_binary") == {
        "fingerprint": "fnv1a64:36c31424e4119557",
        "bytes": 80,
    }
    assert message.get("vertex_pco_opcodes") == {
        "fadd": 0,
        "fmul": 2,
        "mbyp": 3,
        "uvsw_write": 2,
        "uvsw_write_emit_endtask": 0,
        "uvsw_emit_endtask": 1,
    }
    assert message.get("fragment_pco_binary") == {
        "fingerprint": "fnv1a64:0693891931816150",
        "bytes": 184,
    }
    assert message.get("fragment_pco_opcodes") == {
        "fitrp": 1,
        "wdf": 2,
        "smp": 1,
        "mbyp": 18,
    }

    values = message.get("counters")
    assert isinstance(values, dict), f"{case_name}: counters is not an object"
    expected_standard = {
        "ia_vertices": 4,
        "ia_primitives": 2,
        "vs_invocations": 4,
        "gs_invocations": 0,
        "gs_primitives": 0,
        "c_invocations": 2,
        "c_primitives": 2,
        "ps_invocations": 3600,
        "hs_invocations": 0,
        "ds_invocations": 0,
        "cs_invocations": 0,
        "ts_invocations": 0,
        "ms_invocations": 0,
        "ms_primitives": 0,
        "drawlists": 1,
        "setup_triangles": 2,
        "texel_fetches": 32640,
    }
    for field, expected in expected_standard.items():
        assert values.get(field) == expected, (
            f"{case_name}: {field}={values.get(field)!r}, expected {expected}"
        )

    expected_modeled = {
        "virtual_gpu_cycles": 42551,
        "tiler_cycles": 51,
        "renderer_cycles": 42475,
        "usc_groups": 1021,
        "texture_requests": 4080,
        "fifo_stall_events": 0,
        "pool_bytes_in_flight": 0,
        "pool_high_water_bytes": 3108162,
        "vdm_cycles": 9,
        "vertex_fetch_cycles": 5,
        "vertex_attribute_fetches": 8,
        "vertex_attribute_bytes": 64,
        "pco_decode_cycles": 14,
        "pco_instructions": 30,
        "vs_alu_instructions": 20,
        "vs_tex_instructions": 0,
        "vs_memory_instructions": 28,
        "fs_alu_instructions": 77520,
        "fs_tex_instructions": 4080,
        "fs_memory_instructions": 0,
        "usc_slot_cycles": 515,
        "usc_cluster_cycles": 264,
        "clip_cull_cycles": 7,
        "tiler_bin_cycles": 11,
        "parameter_buffer_cycles": 6,
        "parameter_coefficient_sets": 6,
        "parameter_write_bytes": 96,
        "pds_coefficient_tasks": 1020,
        "pds_douti_issues": 2040,
        "usc_coefficient_load_bytes": 48960,
        "tile_scheduler_cycles": 9,
        "isp_cycles": 67,
        "fragment_frontend_cycles": 37,
        "texture_cycles": 8160,
        "pbe_cycles": 44,
        "pixel_data_master_transactions": 1,
        "pixel_data_master_bytes": 16384,
        "pixel_data_master_cycles": 1,
        "tcu_line_accesses": 32640,
        "tcu_read_accesses": 32640,
        "tcu_hits": 32320,
        "tcu_misses": 320,
        "tcu_evictions": 0,
        "tcu_writebacks": 0,
        "tcu_bypassed": 0,
        "tcu_cycles": 32640,
        "slc_line_accesses": 448,
        "slc_read_accesses": 320,
        "slc_write_accesses": 128,
        "slc_hits": 160,
        "slc_misses": 288,
        "slc_evictions": 0,
        "slc_writebacks": 128,
        "slc_bypassed": 0,
        "slc_cycles": 448,
        "dram_read_transactions": 161,
        "dram_write_transactions": 128,
        "dram_read_bytes": 36864,
        "dram_write_bytes": 16384,
        "dram_cycles": 289,
        "memory_direct_read_bytes": 0,
        "memory_direct_write_bytes": 0,
        "framebuffer_dram_readback_bytes": 16384,
        "tiles_binned": 4,
        "tiles_scheduled": 4,
        "covered_pixels": 3600,
        "fragment_candidates": 3600,
        "hsr_rejected_fragments": 0,
        "depth_tested_fragments": 0,
        "depth_rejected_fragments": 0,
        "depth_written_fragments": 0,
        "pbe_color_reads": 0,
        "pbe_blended_fragments": 0,
        "pbe_fragment_writes": 3600,
        "pbe_pixels_written": 4096,
        "functional_frame": 1,
    }
    for field, expected in expected_modeled.items():
        if field in UNIFIED_MEMORY_EXACT_SKIP:
            continue
        assert values.get(field) == expected, (
            f"{case_name}: {field}={values.get(field)!r}, expected {expected}"
        )

    assert message.get("drawlist_stats") == [
        {
            "drawlist": 0,
            "draw_id": 0,
            "vs": {
                "invocations": 4,
                "program": {
                    "groups": 8,
                    "instructions": 8,
                    "alu": 5,
                    "tex": 0,
                    "memory": 3,
                },
                "executed": {"alu": 20, "tex": 0, "memory": 28},
            },
            "fs": {
                "invocations": 3600,
                "program": {
                    "groups": 22,
                    "instructions": 22,
                    "alu": 19,
                    "tex": 1,
                    "memory": 0,
                },
                "executed": {"alu": 77520, "tex": 4080, "memory": 0},
            },
        }
    ]

    assert values["parameter_coefficient_sets"] == values["c_primitives"] * 3
    assert values["parameter_write_bytes"] == (
        values["parameter_coefficient_sets"] * 4 * 4
    )
    assert values["pds_douti_issues"] == values["pds_coefficient_tasks"] * 2
    assert values["usc_coefficient_load_bytes"] == (
        values["pds_coefficient_tasks"] * 12 * 4
    )
    assert values["usc_groups"] == values["pds_coefficient_tasks"] + 1
    assert values["texture_requests"] == values["fs_tex_instructions"]
    assert values["texel_fetches"] == values["texture_requests"] * 8
    assert values["fs_alu_instructions"] == values["texture_requests"] * 19
    assert values["texture_cycles"] >= values["texture_requests"]
    assert values["fragment_candidates"] == (
        values["ps_invocations"] + values["hsr_rejected_fragments"]
    )
    assert values["pbe_fragment_writes"] == values["ps_invocations"]
    assert values["pixel_data_master_bytes"] == values["pbe_pixels_written"] * 4
    assert values["virtual_gpu_cycles"] == (
        values["tiler_cycles"] + values["renderer_cycles"] + 25
    )
    verify_memory_path(
        hello_message,
        message,
        cache_bypass=False,
        framebuffer_bytes=64 * 64 * 4,
    )

    artifact = output_dir / f"{case_name}_sample_000001.png"
    artifact_field = message.get("artifact_png")
    assert isinstance(artifact_field, str)
    assert Path(artifact_field).resolve() == artifact.resolve()
    png_width, png_height, pixels = decode_rgba8_png(artifact)
    assert (png_width, png_height) == (64, 64)
    assert hashlib.sha256(pixels).hexdigest() == (
        "156199bdeca6c5d5f20d69e09a9b145c6f65fbc215e7f04d87f4ba0eb7cf8a15"
    )
    assert [path.name for path in output_dir.rglob("*.png")] == [artifact.name]
    assert not [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name.endswith(".part")
    ]


def verify_fill_tex_trilinear_linear_04_or_05(
    executable: Path, output_dir: Path, case_name: str
) -> None:
    profiles = {
        "fill_tex_trilinear_linear_04": {
            "mode": "systemc-functional-fill-texture-trilinear-linear-04",
            "virtual_time_ns": 26357,
            "ps_invocations": 2304,
            "texel_fetches": 19968,
            "virtual_gpu_cycles": 26381,
            "renderer_cycles": 26305,
            "usc_groups": 625,
            "texture_requests": 2496,
            "pool_high_water_bytes": 2456274,
            "fs_alu_instructions": 47424,
            "usc_slot_cycles": 317,
            "usc_cluster_cycles": 165,
            "pds_coefficient_tasks": 624,
            "pds_douti_issues": 1248,
            "usc_coefficient_load_bytes": 29952,
            "isp_cycles": 46,
            "fragment_frontend_cycles": 25,
            "texture_cycles": 4992,
            "tcu_hits": 19648,
            "decoded_rgba_sha256": (
                "dd2cdfb3e08f66cadf3a509f5156fc2db"
                "0f77d2721a95ee8f48a221f0ddf3795"
            ),
        },
        "fill_tex_trilinear_linear_05": {
            "mode": "systemc-functional-fill-texture-trilinear-linear-05",
            "virtual_time_ns": 26274,
            "ps_invocations": 2116,
            "texel_fetches": 19904,
            "virtual_gpu_cycles": 26298,
            "renderer_cycles": 26222,
            "usc_groups": 623,
            "texture_requests": 2488,
            "pool_high_water_bytes": 2435562,
            "fs_alu_instructions": 47272,
            "usc_slot_cycles": 316,
            "usc_cluster_cycles": 165,
            "pds_coefficient_tasks": 622,
            "pds_douti_issues": 1244,
            "usc_coefficient_load_bytes": 29856,
            "isp_cycles": 44,
            "fragment_frontend_cycles": 25,
            "texture_cycles": 4976,
            "tcu_hits": 19584,
            "decoded_rgba_sha256": (
                "8ae271f3365079dd453d043f14a14a4d"
                "79222a79a2bedf66d5ba8602fe0dd6ee"
            ),
        },
    }
    assert case_name in profiles, f"unsupported trilinear profile {case_name}"
    profile = profiles[case_name]
    completed = invoke_case(
        executable,
        case_name,
        output_dir,
        frames=1,
        cache_bypass="off",
        width=64,
        height=64,
    )
    assert completed.returncode == 0, (
        f"{case_name} model run failed:\n" + process_details(completed)
    )

    messages = json_messages(completed)
    hello = [message for message in messages if message.get("type") == "hello"]
    counters = [
        message for message in messages if message.get("type") == "counter"
    ]
    done = [message for message in messages if message.get("type") == "done"]
    assert len(hello) == len(counters) == len(done) == 1

    hello_message = hello[0]
    expected_hello = {
        "backend": "pvrgpu",
        "mode": profile["mode"],
        "functional_scope": f"{case_name}-pco-iss-v1",
        "command_source": "builtin-glbench-fixture",
        "shader_binary": "mesa-pco-public-encoding",
        "pco_subset": "fmul-fitrp-wdf-smp-mbyp-uvsw-texture",
        "reference_uarch": "pvrgpu-ref-v1",
        "uarch_provenance": "assumed",
        "timing_provenance": "uncalibrated",
        "cache_bypass": False,
        "cache_policy": "set-associative-write-back-write-allocate-true-lru",
        "framebuffer_source": "dram-readback",
        "dram_fixed_latency_cycles": 1,
        "tile_width": 32,
        "tile_height": 32,
        "workload": case_name,
    }
    for field, expected in expected_hello.items():
        assert hello_message.get(field) == expected, (
            f"{case_name}: hello {field}={hello_message.get(field)!r}, "
            f"expected {expected!r}"
        )
    assert hello_message.get("tcu_cache") == {
        "capacity_bytes": 24 * 1024,
        "line_bytes": 64,
        "ways": 4,
        "banks": 4,
    }
    assert hello_message.get("slc_cache") == {
        "capacity_bytes": 2 * 1024 * 1024,
        "line_bytes": 128,
        "ways": 8,
        "banks": 8,
    }

    done_message = done[0]
    assert done_message.get("pool_leaks") == 0
    assert done_message.get("pool_bytes_in_flight") == 0
    assert done_message.get("pool_allocations") == done_message.get(
        "pool_releases"
    )

    message = counters[0]
    expected_provenance = {
        "backend": "pvrgpu",
        "source": "pvrgpu-systemc",
        "provenance": "modeled",
        "functional_scope": f"{case_name}-pco-iss-v1",
        "command_source": "builtin-glbench-fixture",
        "timing_provenance": "uncalibrated",
        "cache_bypass": False,
        "framebuffer_source": "dram-readback",
        "frame": 1,
        "marker": case_name,
        "virtual_time_ns": profile["virtual_time_ns"],
    }
    for field, expected in expected_provenance.items():
        if field == "virtual_time_ns":
            continue
        assert message.get(field) == expected, (
            f"{case_name}: counter {field}={message.get(field)!r}, "
            f"expected {expected!r}"
        )
    assert message.get("vertex_pco_binary") == {
        "fingerprint": "fnv1a64:36c31424e4119557",
        "bytes": 80,
    }
    assert message.get("vertex_pco_opcodes") == {
        "fadd": 0,
        "fmul": 2,
        "mbyp": 3,
        "uvsw_write": 2,
        "uvsw_write_emit_endtask": 0,
        "uvsw_emit_endtask": 1,
    }
    assert message.get("fragment_pco_binary") == {
        "fingerprint": "fnv1a64:0693891931816150",
        "bytes": 184,
    }
    assert message.get("fragment_pco_opcodes") == {
        "fitrp": 1,
        "wdf": 2,
        "smp": 1,
        "mbyp": 18,
    }

    values = message.get("counters")
    assert isinstance(values, dict), f"{case_name}: counters is not an object"
    expected_values = {
        "ia_vertices": 4,
        "ia_primitives": 2,
        "vs_invocations": 4,
        "gs_invocations": 0,
        "gs_primitives": 0,
        "c_invocations": 2,
        "c_primitives": 2,
        "ps_invocations": profile["ps_invocations"],
        "hs_invocations": 0,
        "ds_invocations": 0,
        "cs_invocations": 0,
        "ts_invocations": 0,
        "ms_invocations": 0,
        "ms_primitives": 0,
        "drawlists": 1,
        "setup_triangles": 2,
        "texel_fetches": profile["texel_fetches"],
        "virtual_gpu_cycles": profile["virtual_gpu_cycles"],
        "tiler_cycles": 51,
        "renderer_cycles": profile["renderer_cycles"],
        "usc_groups": profile["usc_groups"],
        "texture_requests": profile["texture_requests"],
        "fifo_stall_events": 0,
        "pool_bytes_in_flight": 0,
        "pool_high_water_bytes": profile["pool_high_water_bytes"],
        "vdm_cycles": 9,
        "vertex_fetch_cycles": 5,
        "vertex_attribute_fetches": 8,
        "vertex_attribute_bytes": 64,
        "pco_decode_cycles": 14,
        "pco_instructions": 30,
        "vs_alu_instructions": 20,
        "vs_tex_instructions": 0,
        "vs_memory_instructions": 28,
        "fs_alu_instructions": profile["fs_alu_instructions"],
        "fs_tex_instructions": profile["texture_requests"],
        "fs_memory_instructions": 0,
        "usc_slot_cycles": profile["usc_slot_cycles"],
        "usc_cluster_cycles": profile["usc_cluster_cycles"],
        "clip_cull_cycles": 7,
        "tiler_bin_cycles": 11,
        "parameter_buffer_cycles": 6,
        "parameter_coefficient_sets": 6,
        "parameter_write_bytes": 96,
        "pds_coefficient_tasks": profile["pds_coefficient_tasks"],
        "pds_douti_issues": profile["pds_douti_issues"],
        "usc_coefficient_load_bytes": profile["usc_coefficient_load_bytes"],
        "tile_scheduler_cycles": 9,
        "isp_cycles": profile["isp_cycles"],
        "fragment_frontend_cycles": profile["fragment_frontend_cycles"],
        "texture_cycles": profile["texture_cycles"],
        "pbe_cycles": 44,
        "pixel_data_master_transactions": 1,
        "pixel_data_master_bytes": 16384,
        "pixel_data_master_cycles": 1,
        "tcu_line_accesses": profile["texel_fetches"],
        "tcu_read_accesses": profile["texel_fetches"],
        "tcu_hits": profile["tcu_hits"],
        "tcu_misses": 320,
        "tcu_evictions": 0,
        "tcu_writebacks": 0,
        "tcu_bypassed": 0,
        "tcu_cycles": profile["texel_fetches"],
        "slc_line_accesses": 448,
        "slc_read_accesses": 320,
        "slc_write_accesses": 128,
        "slc_hits": 160,
        "slc_misses": 288,
        "slc_evictions": 0,
        "slc_writebacks": 128,
        "slc_bypassed": 0,
        "slc_cycles": 448,
        "dram_read_transactions": 161,
        "dram_write_transactions": 128,
        "dram_read_bytes": 36864,
        "dram_write_bytes": 16384,
        "dram_cycles": 289,
        "memory_direct_read_bytes": 0,
        "memory_direct_write_bytes": 0,
        "framebuffer_dram_readback_bytes": 16384,
        "tiles_binned": 4,
        "tiles_scheduled": 4,
        "covered_pixels": profile["ps_invocations"],
        "fragment_candidates": profile["ps_invocations"],
        "hsr_rejected_fragments": 0,
        "depth_tested_fragments": 0,
        "depth_rejected_fragments": 0,
        "depth_written_fragments": 0,
        "pbe_color_reads": 0,
        "pbe_blended_fragments": 0,
        "pbe_fragment_writes": profile["ps_invocations"],
        "pbe_pixels_written": 4096,
        "functional_frame": 1,
    }
    for field, expected in expected_values.items():
        if field in UNIFIED_MEMORY_EXACT_SKIP:
            continue
        assert values.get(field) == expected, (
            f"{case_name}: {field}={values.get(field)!r}, expected {expected}"
        )
    assert set(values) == set(expected_values), (
        f"{case_name}: counter fields={sorted(values)}, "
        f"expected {sorted(expected_values)}"
    )

    assert message.get("drawlist_stats") == [
        {
            "drawlist": 0,
            "draw_id": 0,
            "vs": {
                "invocations": 4,
                "program": {
                    "groups": 8,
                    "instructions": 8,
                    "alu": 5,
                    "tex": 0,
                    "memory": 3,
                },
                "executed": {"alu": 20, "tex": 0, "memory": 28},
            },
            "fs": {
                "invocations": profile["ps_invocations"],
                "program": {
                    "groups": 22,
                    "instructions": 22,
                    "alu": 19,
                    "tex": 1,
                    "memory": 0,
                },
                "executed": {
                    "alu": profile["fs_alu_instructions"],
                    "tex": profile["texture_requests"],
                    "memory": 0,
                },
            },
        }
    ]

    assert values["texture_requests"] == values["fs_tex_instructions"]
    assert values["texel_fetches"] == values["texture_requests"] * 8
    assert values["texture_cycles"] >= values["texture_requests"]
    assert values["fragment_candidates"] == values["ps_invocations"]
    assert values["pbe_fragment_writes"] == values["ps_invocations"]
    assert values["virtual_gpu_cycles"] == (
        values["tiler_cycles"] + values["renderer_cycles"] + 25
    )
    verify_memory_path(
        hello_message,
        message,
        cache_bypass=False,
        framebuffer_bytes=64 * 64 * 4,
    )

    artifact = output_dir / f"{case_name}_sample_000001.png"
    artifact_field = message.get("artifact_png")
    assert isinstance(artifact_field, str)
    assert Path(artifact_field).resolve() == artifact.resolve()
    png_width, png_height, pixels = decode_rgba8_png(artifact)
    assert (png_width, png_height) == (64, 64)
    assert hashlib.sha256(pixels).hexdigest() == profile[
        "decoded_rgba_sha256"
    ]
    assert [path.name for path in output_dir.rglob("*.png")] == [artifact.name]
    assert not [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name.endswith(".part")
    ]
def verify_cache_bypass_options(executable: Path) -> None:
    help_result = subprocess.run(
        [str(executable), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert help_result.returncode == 0, process_details(help_result)
    assert "--memory-mode direct|bypass|cache" in help_result.stdout
    assert "--cache-bypass on|off" in help_result.stdout
    assert "Run SLC/DRAM simulation (default)" in help_result.stdout

    invalid = subprocess.run(
        [str(executable), "--cache-bypass", "enabled"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert invalid.returncode != 0, "invalid cache bypass value was accepted"
    assert "expected on or off" in invalid.stderr

    missing = subprocess.run(
        [str(executable), "--cache-bypass"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert missing.returncode != 0, "missing cache bypass value was accepted"
    assert "Missing value after --cache-bypass" in missing.stderr


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} PVRGPU_MODEL", file=sys.stderr)
        return 2
    executable = Path(sys.argv[1]).resolve()
    verify_cache_bypass_options(executable)
    with tempfile.TemporaryDirectory(prefix="pvrgpu-functional-test-") as temporary:
        root = Path(temporary)
        verify_driver_clear_command(executable, root / "driver-clear-green")
        verify_driver_triangle_command(executable, root / "driver-triangle-red")
        verify_driver_indexed_quad_framebuffer_size(
            executable, root / "driver-indexed-quad-framebuffer-size"
        )
        verify_driver_primitive_sequence_is_rejected(
            executable, root / "driver-primitive-sequence"
        )
        verify_fill_solid(executable, root / "fill-solid")
        verify_fill_solid_bypass(
            executable,
            root / "fill-solid-bypass",
            b"\xFF\x00\x00\xFF" * PIXEL_COUNT,
        )
        verify_fill_solid_direct(
            executable,
            root / "fill-solid-direct",
            b"\xFF\x00\x00\xFF" * PIXEL_COUNT,
        )
        verify_fill_solid_blended(
            executable,
            root / "fill-solid-blended",
        )
        verify_fill_depth_case(
            executable,
            "fill_solid_depth_neq",
            root / "fill-solid-depth-neq",
            expected_ps_invocations=PIXEL_COUNT,
            expected_depth_rejected=0,
            expected_pixel=b"\xFF\x00\x00\xFF",
            cache_bypass="off",
        )
        verify_fill_depth_case(
            executable,
            "fill_solid_depth_never",
            root / "fill-solid-depth-never",
            expected_ps_invocations=0,
            expected_depth_rejected=PIXEL_COUNT,
            expected_pixel=b"\x00\x00\x00\xFF",
            cache_bypass="on",
        )
        verify_triangle_setup(executable, root / "triangle-setup")
        verify_triangle_setup_all_culled(
            executable, root / "triangle-setup-all-culled"
        )
        verify_triangle_setup_half_culled(
            executable, root / "triangle-setup-half-culled"
        )
        verify_attribute_fetch_shader(
            executable, root / "attribute-fetch-shader"
        )
        verify_attribute_fetch_shader(
            executable,
            root / "attribute-fetch-shader-2-attr",
            "attribute_fetch_shader_2_attr",
        )
        verify_attribute_fetch_shader(
            executable,
            root / "attribute-fetch-shader-4-attr",
            "attribute_fetch_shader_4_attr",
        )
        verify_attribute_fetch_shader(
            executable,
            root / "attribute-fetch-shader-8-attr",
            "attribute_fetch_shader_8_attr",
        )
        verify_varyings_shader_one(
            executable, root / "varyings-shader-one"
        )
        verify_varyings_shader_two(
            executable, root / "varyings-shader-two"
        )
        verify_varyings_shader_four(
            executable, root / "varyings-shader-four"
        )
        verify_varyings_shader_eight(
            executable, root / "varyings-shader-eight"
        )
        verify_fill_tex_nearest(
            executable, root / "fill-tex-nearest"
        )
        verify_fill_tex_bilinear(
            executable, root / "fill-tex-bilinear"
        )
        verify_fill_tex_trilinear_linear_01(
            executable, root / "fill-tex-trilinear-linear-01"
        )
        verify_fill_tex_trilinear_linear_04_or_05(
            executable,
            root / "fill-tex-trilinear-linear-04",
            "fill_tex_trilinear_linear_04",
        )
        verify_fill_tex_trilinear_linear_04_or_05(
            executable,
            root / "fill-tex-trilinear-linear-05",
            "fill_tex_trilinear_linear_05",
        )
    print("SystemC functional raster state-case integration: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
