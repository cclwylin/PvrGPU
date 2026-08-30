#!/usr/bin/env python3
"""Fail-fast counter/state gate for llvmpipe versus PvrGPU GLBench runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from counter_protocol import (
    counter_record_from_message,
    parse_jsonl_line,
    parse_markdown_report,
)


PIPELINE_FIELDS = (
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
)


def one(messages: list[dict[str, object]], message_type: str) -> dict[str, object]:
    selected = [message for message in messages if message.get("type") == message_type]
    if len(selected) != 1:
        raise AssertionError(
            f"expected one {message_type} message, found {len(selected)}"
        )
    return selected[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--golden-report", type=Path, required=True)
    parser.add_argument("--model-log", type=Path, required=True)
    options = parser.parse_args()

    golden_report = parse_markdown_report(options.golden_report)
    if "llvmpipe" not in golden_report.metadata.get("Renderer", "").lower():
        raise AssertionError("Golden report renderer is not llvmpipe")
    if len(golden_report.records) != 1:
        raise AssertionError("Golden report must contain exactly one captured frame")
    golden = golden_report.records[0]

    messages = [
        parse_jsonl_line(line)
        for line in options.model_log.read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith("{")
    ]
    hello = one(messages, "hello")
    model_message = one(messages, "counter")
    done = one(messages, "done")
    model = counter_record_from_message(model_message)
    if model.provenance != "modeled":
        raise AssertionError(f"PvrGPU counter provenance is {model.provenance!r}")
    if done.get("pool_leaks") != 0 or done.get("pool_bytes_in_flight") != 0:
        raise AssertionError("PvrGPU MemoryPool did not reach zero live bytes")

    for field in PIPELINE_FIELDS:
        golden_value = golden.values.get(field, 0)
        model_value = model.values.get(field, 0)
        if model_value != golden_value:
            raise AssertionError(
                f"{options.case}: {field} PvrGPU={model_value}, "
                f"llvmpipe={golden_value}"
            )

    ps_invocations = int(model.values["ps_invocations"])
    if len(model.drawlist_stats) != 1:
        raise AssertionError("PvrGPU must report exactly one DrawList")
    drawlist = model.drawlist_stats[0]
    if drawlist.vertex.invocations != int(model.values["vs_invocations"]):
        raise AssertionError("DrawList VS invocation total does not match frame total")
    if drawlist.fragment.invocations != ps_invocations:
        raise AssertionError("DrawList FS invocation total does not match frame total")

    attribute_counts = {
        "attribute_fetch_shader": 1,
        "attribute_fetch_shader_2_attr": 2,
        "attribute_fetch_shader_4_attr": 4,
        "attribute_fetch_shader_8_attr": 8,
    }
    if options.case in attribute_counts:
        attribute_count = attribute_counts[options.case]
        vs_invocations = int(model.values["vs_invocations"])
        expected_fetches = vs_invocations * attribute_count
        expected_bytes = expected_fetches * 2 * 4
        expected_vertex_fetch_cycles = {1: 25, 2: 46, 4: 88, 8: 171}[
            attribute_count
        ]
        expected_vertex_alu = {1: 4, 2: 4, 4: 8, 8: 16}[attribute_count]
        expected_program_groups = {1: 6, 2: 6, 4: 10, 8: 18}[
            attribute_count
        ]
        expected_values = {
            "vertex_attribute_fetches": expected_fetches,
            "vertex_attribute_bytes": expected_bytes,
            "vertex_fetch_cycles": expected_vertex_fetch_cycles,
            "pco_instructions": {1: 10, 2: 10, 4: 14, 8: 22}[
                attribute_count
            ],
            "pco_decode_cycles": {1: 9, 2: 9, 4: 10, 8: 12}[
                attribute_count
            ],
            "tiler_cycles": {1: 1367, 2: 1388, 4: 1431, 8: 1516}[
                attribute_count
            ],
            "renderer_cycles": 330,
            "virtual_gpu_cycles": {1: 1722, 2: 1743, 4: 1786, 8: 1871}[
                attribute_count
            ],
            "vs_alu_instructions": vs_invocations * expected_vertex_alu,
            "vs_tex_instructions": 0,
            "vs_memory_instructions": vs_invocations * 5,
            "fs_alu_instructions": 0,
            "fs_tex_instructions": 0,
            "fs_memory_instructions": 0,
        }
        for field, expected in expected_values.items():
            actual = model.values.get(field)
            if actual != expected:
                raise AssertionError(
                    f"{options.case}: {field} PvrGPU={actual}, expected {expected}"
                )
        expected_vertex_program = (
            expected_program_groups,
            expected_program_groups,
            expected_vertex_alu,
            0,
            2,
        )
        actual_vertex_program = (
            drawlist.vertex.program_groups,
            drawlist.vertex.program_instructions,
            drawlist.vertex.program_alu_instructions,
            drawlist.vertex.program_tex_instructions,
            drawlist.vertex.program_memory_instructions,
        )
        if actual_vertex_program != expected_vertex_program:
            raise AssertionError(
                f"{options.case}: VS static program {actual_vertex_program}, "
                f"expected {expected_vertex_program}"
            )
        expected_vertex_executed = (
            vs_invocations * expected_vertex_alu,
            0,
            vs_invocations * 5,
        )
        actual_vertex_executed = (
            drawlist.vertex.executed_alu_instructions,
            drawlist.vertex.executed_tex_instructions,
            drawlist.vertex.executed_memory_instructions,
        )
        if actual_vertex_executed != expected_vertex_executed:
            raise AssertionError(
                f"{options.case}: VS executed {actual_vertex_executed}, "
                f"expected {expected_vertex_executed}"
            )
        if hello.get("functional_scope") != f"{options.case}-pco-iss-v1":
            raise AssertionError("PvrGPU functional scope does not match the case")
        if hello.get("shader_binary") != "mesa-pco-public-encoding":
            raise AssertionError("PvrGPU did not identify the public PCO binary")
        expected_program_evidence = {
            "attribute_fetch_shader": {
                "subset": "mbyp-uvsw-attribute-fetch",
                "binary": {
                    "fingerprint": "fnv1a64:48cf8717db7aa8cf",
                    "bytes": 56,
                },
                "opcodes": {
                    "fadd": 0,
                    "mbyp": 4,
                    "uvsw_write": 1,
                    "uvsw_write_emit_endtask": 0,
                    "uvsw_emit_endtask": 1,
                },
            },
            "attribute_fetch_shader_2_attr": {
                "subset": "fadd-mbyp-uvsw-attribute-fetch",
                "binary": {
                    "fingerprint": "fnv1a64:4fb7f3aba4b44c19",
                    "bytes": 56,
                },
                "opcodes": {
                    "fadd": 2,
                    "mbyp": 2,
                    "uvsw_write": 1,
                    "uvsw_write_emit_endtask": 0,
                    "uvsw_emit_endtask": 1,
                },
            },
            "attribute_fetch_shader_4_attr": {
                "subset": "fadd-mbyp-uvsw-attribute-fetch",
                "binary": {
                    "fingerprint": "fnv1a64:c54ea51cdaab08a0",
                    "bytes": 96,
                },
                "opcodes": {
                    "fadd": 6,
                    "mbyp": 2,
                    "uvsw_write": 1,
                    "uvsw_write_emit_endtask": 0,
                    "uvsw_emit_endtask": 1,
                },
            },
            "attribute_fetch_shader_8_attr": {
                "subset": "fadd-mbyp-uvsw-attribute-fetch",
                "binary": {
                    "fingerprint": "fnv1a64:87d4d7b5e46ff241",
                    "bytes": 176,
                },
                "opcodes": {
                    "fadd": 14,
                    "mbyp": 2,
                    "uvsw_write": 1,
                    "uvsw_write_emit_endtask": 0,
                    "uvsw_emit_endtask": 1,
                },
            },
        }[options.case]
        if hello.get("pco_subset") != expected_program_evidence["subset"]:
            raise AssertionError("PvrGPU hello PCO subset does not match the case")
        if model_message.get("vertex_pco_binary") != expected_program_evidence[
            "binary"
        ]:
            raise AssertionError("actual vertex PCO binary fingerprint mismatch")
        if model_message.get("vertex_pco_opcodes") != expected_program_evidence[
            "opcodes"
        ]:
            raise AssertionError("actual decoded vertex PCO opcode histogram mismatch")

    if options.case == "varyings_shader_1":
        expected_standard = {
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
        }
        for field, expected in expected_standard.items():
            golden_value = golden.values.get(field, 0)
            model_value = model.values.get(field, 0)
            if golden_value != expected or model_value != expected:
                raise AssertionError(
                    f"{options.case}: {field} llvmpipe={golden_value}, "
                    f"PvrGPU={model_value}, expected {expected}"
                )

        expected_hello = {
            "mode": "systemc-functional-varyings-shader",
            "functional_scope": "varyings_shader_1-pco-iss-v1",
            "pco_subset": "fitrp-wdf-mbyp-uvsw-varying",
            "shader_binary": "mesa-pco-public-encoding",
            "cache_bypass": False,
            "framebuffer_source": "dram-readback",
            "dram_fixed_latency_cycles": 1,
            "tile_width": 32,
            "tile_height": 32,
        }
        for field, expected in expected_hello.items():
            if hello.get(field) != expected:
                raise AssertionError(
                    f"{options.case}: hello {field}={hello.get(field)!r}, "
                    f"expected {expected!r}"
                )

        expected_program_evidence = {
            "vertex_pco_binary": {
                "fingerprint": "fnv1a64:c1a8f8a4f58fc81f",
                "bytes": 72,
            },
            "vertex_pco_opcodes": {
                "fadd": 0,
                "mbyp": 4,
                "uvsw_write": 2,
                "uvsw_write_emit_endtask": 0,
                "uvsw_emit_endtask": 1,
            },
            "fragment_pco_binary": {
                "fingerprint": "fnv1a64:76458bbcec6f53bf",
                "bytes": 48,
            },
            "fragment_pco_opcodes": {
                "fitrp": 1,
                "wdf": 1,
                "mbyp": 4,
            },
        }
        for field, expected in expected_program_evidence.items():
            if model_message.get(field) != expected:
                raise AssertionError(
                    f"{options.case}: {field}={model_message.get(field)!r}, "
                    f"expected {expected!r}"
                )

        expected_modeled = {
            "virtual_gpu_cycles": 1379,
            "tiler_cycles": 55,
            "renderer_cycles": 1299,
            "usc_groups": 1159,
            "texture_requests": 0,
            "vdm_cycles": 9,
            "vertex_fetch_cycles": 5,
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
            "usc_slot_cycles": 584,
            "usc_cluster_cycles": 298,
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
        }
        for field, expected in expected_modeled.items():
            actual = model.values.get(field)
            if actual != expected:
                raise AssertionError(
                    f"{options.case}: {field} PvrGPU={actual}, "
                    f"expected {expected}"
                )

        expected_vertex_program = (7, 7, 4, 0, 3)
        actual_vertex_program = (
            drawlist.vertex.program_groups,
            drawlist.vertex.program_instructions,
            drawlist.vertex.program_alu_instructions,
            drawlist.vertex.program_tex_instructions,
            drawlist.vertex.program_memory_instructions,
        )
        expected_vertex_executed = (100, 0, 225)
        actual_vertex_executed = (
            drawlist.vertex.executed_alu_instructions,
            drawlist.vertex.executed_tex_instructions,
            drawlist.vertex.executed_memory_instructions,
        )
        expected_fragment_program = (6, 6, 5, 0, 0)
        actual_fragment_program = (
            drawlist.fragment.program_groups,
            drawlist.fragment.program_instructions,
            drawlist.fragment.program_alu_instructions,
            drawlist.fragment.program_tex_instructions,
            drawlist.fragment.program_memory_instructions,
        )
        expected_fragment_executed = (20480, 0, 0)
        actual_fragment_executed = (
            drawlist.fragment.executed_alu_instructions,
            drawlist.fragment.executed_tex_instructions,
            drawlist.fragment.executed_memory_instructions,
        )
        actual_drawlist = (
            drawlist.drawlist,
            drawlist.draw_id,
            drawlist.vertex.invocations,
            actual_vertex_program,
            actual_vertex_executed,
            drawlist.fragment.invocations,
            actual_fragment_program,
            actual_fragment_executed,
        )
        expected_drawlist = (
            0,
            0,
            25,
            expected_vertex_program,
            expected_vertex_executed,
            4096,
            expected_fragment_program,
            expected_fragment_executed,
        )
        if actual_drawlist != expected_drawlist:
            raise AssertionError(
                f"{options.case}: DrawList {actual_drawlist}, "
                f"expected {expected_drawlist}"
            )

        values = model.values
        relations = (
            (
                values["parameter_coefficient_sets"],
                values["c_primitives"] * 5,
                "five coefficient sets per primitive",
            ),
            (
                values["parameter_write_bytes"],
                values["parameter_coefficient_sets"] * 4 * 4,
                "four dwords per coefficient set",
            ),
            (
                values["pds_douti_issues"],
                values["pds_coefficient_tasks"] * 2,
                "two DOUTI issues per fragment-quad task",
            ),
            (
                values["usc_coefficient_load_bytes"],
                values["pds_coefficient_tasks"] * 20 * 4,
                "twenty coefficient dwords per USC task",
            ),
            (
                values["usc_groups"],
                values["pds_coefficient_tasks"] + 7,
                "fragment tasks plus seven vertex program groups",
            ),
            (
                values["fragment_candidates"],
                values["ps_invocations"] + values["hsr_rejected_fragments"],
                "HSR candidate conservation",
            ),
            (
                values["pbe_fragment_writes"],
                values["ps_invocations"],
                "one PBE fragment write per executed fragment",
            ),
            (
                values["pixel_data_master_bytes"],
                values["pbe_pixels_written"] * 4,
                "RGBA8 PixelDM payload",
            ),
            (
                values["slc_line_accesses"],
                values["pixel_data_master_bytes"] // 128,
                "128-byte SLC line coverage",
            ),
            (
                values["dram_cycles"],
                values["dram_read_transactions"]
                + values["dram_write_transactions"],
                "one fixed-latency cycle per DRAM transaction",
            ),
            (
                values["virtual_gpu_cycles"],
                values["tiler_cycles"]
                + values["renderer_cycles"]
                + 25,
                "frame critical-path composition",
            ),
        )
        for actual, expected, description in relations:
            if actual != expected:
                raise AssertionError(
                    f"{options.case}: {description}: {actual} != {expected}"
                )

    if options.case == "varyings_shader_2":
        expected_standard = {
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
        }
        for field, expected in expected_standard.items():
            golden_value = golden.values.get(field, 0)
            model_value = model.values.get(field, 0)
            if golden_value != expected or model_value != expected:
                raise AssertionError(
                    f"{options.case}: {field} llvmpipe={golden_value}, "
                    f"PvrGPU={model_value}, expected {expected}"
                )

        expected_hello = {
            "mode": "systemc-functional-varyings-shader",
            "functional_scope": "varyings_shader_2-pco-iss-v1",
            "pco_subset": "fmul-fitrp-wdf-fadd-mbyp-uvsw-varying",
            "shader_binary": "mesa-pco-public-encoding",
            "cache_bypass": False,
            "framebuffer_source": "dram-readback",
            "dram_fixed_latency_cycles": 1,
            "tile_width": 32,
            "tile_height": 32,
        }
        for field, expected in expected_hello.items():
            if hello.get(field) != expected:
                raise AssertionError(
                    f"{options.case}: hello {field}={hello.get(field)!r}, "
                    f"expected {expected!r}"
                )

        expected_program_evidence = {
            "vertex_pco_binary": {
                "fingerprint": "fnv1a64:ffd408e5a8ae5f7c",
                "bytes": 120,
            },
            "vertex_pco_opcodes": {
                "fadd": 0,
                "fmul": 2,
                "mbyp": 6,
                "uvsw_write": 3,
                "uvsw_write_emit_endtask": 0,
                "uvsw_emit_endtask": 1,
            },
            "fragment_pco_binary": {
                "fingerprint": "fnv1a64:9c2a8c68ef09d5d1",
                "bytes": 104,
            },
            "fragment_pco_opcodes": {
                "fitrp": 2,
                "wdf": 2,
                "fadd": 4,
                "mbyp": 4,
            },
        }
        for field, expected in expected_program_evidence.items():
            if model_message.get(field) != expected:
                raise AssertionError(
                    f"{options.case}: {field}={model_message.get(field)!r}, "
                    f"expected {expected!r}"
                )

        expected_modeled = {
            "virtual_gpu_cycles": 1381,
            "tiler_cycles": 56,
            "renderer_cycles": 1300,
            "usc_groups": 1159,
            "texture_requests": 0,
            "vdm_cycles": 9,
            "vertex_fetch_cycles": 5,
            "vertex_attribute_fetches": 25,
            "vertex_attribute_bytes": 200,
            "pco_decode_cycles": 12,
            "pco_instructions": 24,
            "vs_alu_instructions": 200,
            "vs_tex_instructions": 0,
            "vs_memory_instructions": 325,
            "fs_alu_instructions": 40960,
            "fs_tex_instructions": 0,
            "fs_memory_instructions": 0,
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
        }
        for field, expected in expected_modeled.items():
            actual = model.values.get(field)
            if actual != expected:
                raise AssertionError(
                    f"{options.case}: {field} PvrGPU={actual}, "
                    f"expected {expected}"
                )

        expected_vertex_program = (12, 12, 8, 0, 4)
        actual_vertex_program = (
            drawlist.vertex.program_groups,
            drawlist.vertex.program_instructions,
            drawlist.vertex.program_alu_instructions,
            drawlist.vertex.program_tex_instructions,
            drawlist.vertex.program_memory_instructions,
        )
        expected_vertex_executed = (200, 0, 325)
        actual_vertex_executed = (
            drawlist.vertex.executed_alu_instructions,
            drawlist.vertex.executed_tex_instructions,
            drawlist.vertex.executed_memory_instructions,
        )
        expected_fragment_program = (12, 12, 10, 0, 0)
        actual_fragment_program = (
            drawlist.fragment.program_groups,
            drawlist.fragment.program_instructions,
            drawlist.fragment.program_alu_instructions,
            drawlist.fragment.program_tex_instructions,
            drawlist.fragment.program_memory_instructions,
        )
        expected_fragment_executed = (40960, 0, 0)
        actual_fragment_executed = (
            drawlist.fragment.executed_alu_instructions,
            drawlist.fragment.executed_tex_instructions,
            drawlist.fragment.executed_memory_instructions,
        )
        actual_drawlist = (
            drawlist.drawlist,
            drawlist.draw_id,
            drawlist.vertex.invocations,
            actual_vertex_program,
            actual_vertex_executed,
            drawlist.fragment.invocations,
            actual_fragment_program,
            actual_fragment_executed,
        )
        expected_drawlist = (
            0,
            0,
            25,
            expected_vertex_program,
            expected_vertex_executed,
            4096,
            expected_fragment_program,
            expected_fragment_executed,
        )
        if actual_drawlist != expected_drawlist:
            raise AssertionError(
                f"{options.case}: DrawList {actual_drawlist}, "
                f"expected {expected_drawlist}"
            )

        values = model.values
        relations = (
            (
                values["parameter_coefficient_sets"],
                values["c_primitives"] * 9,
                "nine coefficient sets per primitive",
            ),
            (
                values["parameter_write_bytes"],
                values["parameter_coefficient_sets"] * 4 * 4,
                "four dwords per coefficient set",
            ),
            (
                values["pds_douti_issues"],
                values["pds_coefficient_tasks"] * 2,
                "two DOUTI issues per fragment-quad task",
            ),
            (
                values["usc_coefficient_load_bytes"],
                values["pds_coefficient_tasks"] * 36 * 4,
                "thirty-six coefficient dwords per USC task",
            ),
            (
                values["usc_groups"],
                values["pds_coefficient_tasks"] + 7,
                "fragment tasks plus seven vertex issue groups",
            ),
            (
                values["fragment_candidates"],
                values["ps_invocations"] + values["hsr_rejected_fragments"],
                "HSR candidate conservation",
            ),
            (
                values["pbe_fragment_writes"],
                values["ps_invocations"],
                "one PBE fragment write per executed fragment",
            ),
            (
                values["pixel_data_master_bytes"],
                values["pbe_pixels_written"] * 4,
                "RGBA8 PixelDM payload",
            ),
            (
                values["slc_line_accesses"],
                values["pixel_data_master_bytes"] // 128,
                "128-byte SLC line coverage",
            ),
            (
                values["dram_cycles"],
                values["dram_read_transactions"]
                + values["dram_write_transactions"],
                "one fixed-latency cycle per DRAM transaction",
            ),
            (
                values["virtual_gpu_cycles"],
                values["tiler_cycles"]
                + values["renderer_cycles"]
                + 25,
                "frame critical-path composition",
            ),
        )
        for actual, expected, description in relations:
            if actual != expected:
                raise AssertionError(
                    f"{options.case}: {description}: {actual} != {expected}"
                )

    if options.case == "varyings_shader_4":
        expected_standard = {
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
        }
        for field, expected in expected_standard.items():
            golden_value = golden.values.get(field, 0)
            model_value = model.values.get(field, 0)
            if golden_value != expected or model_value != expected:
                raise AssertionError(
                    f"{options.case}: {field} llvmpipe={golden_value}, "
                    f"PvrGPU={model_value}, expected {expected}"
                )

        expected_hello = {
            "mode": "systemc-functional-varyings-shader",
            "functional_scope": "varyings_shader_4-pco-iss-v1",
            "pco_subset": "fmul-fitrp-wdf-fadd-mbyp-uvsw-varying",
            "shader_binary": "mesa-pco-public-encoding",
            "cache_bypass": False,
            "framebuffer_source": "dram-readback",
            "dram_fixed_latency_cycles": 1,
            "tile_width": 32,
            "tile_height": 32,
        }
        for field, expected in expected_hello.items():
            if hello.get(field) != expected:
                raise AssertionError(
                    f"{options.case}: hello {field}={hello.get(field)!r}, "
                    f"expected {expected!r}"
                )

        expected_program_evidence = {
            "vertex_pco_binary": {
                "fingerprint": "fnv1a64:a654c923dfea45ef",
                "bytes": 136,
            },
            "vertex_pco_opcodes": {
                "fadd": 0,
                "fmul": 2,
                "mbyp": 6,
                "uvsw_write": 5,
                "uvsw_write_emit_endtask": 0,
                "uvsw_emit_endtask": 1,
            },
            "fragment_pco_binary": {
                "fingerprint": "fnv1a64:1dbf7c0552b6b385",
                "bytes": 216,
            },
            "fragment_pco_opcodes": {
                "fitrp": 4,
                "wdf": 4,
                "fadd": 12,
                "mbyp": 4,
            },
        }
        for field, expected in expected_program_evidence.items():
            if model_message.get(field) != expected:
                raise AssertionError(
                    f"{options.case}: {field}={model_message.get(field)!r}, "
                    f"expected {expected!r}"
                )

        expected_modeled = {
            "virtual_gpu_cycles": 1385,
            "tiler_cycles": 57,
            "renderer_cycles": 1303,
            "usc_groups": 1159,
            "texture_requests": 0,
            "fifo_stall_events": 0,
            "pool_bytes_in_flight": 0,
            "pool_high_water_bytes": 1037240,
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
        for field, expected in expected_modeled.items():
            actual = model.values.get(field)
            if actual != expected:
                raise AssertionError(
                    f"{options.case}: {field} PvrGPU={actual}, "
                    f"expected {expected}"
                )

        expected_vertex_program = (14, 14, 8, 0, 6)
        actual_vertex_program = (
            drawlist.vertex.program_groups,
            drawlist.vertex.program_instructions,
            drawlist.vertex.program_alu_instructions,
            drawlist.vertex.program_tex_instructions,
            drawlist.vertex.program_memory_instructions,
        )
        expected_vertex_executed = (200, 0, 525)
        actual_vertex_executed = (
            drawlist.vertex.executed_alu_instructions,
            drawlist.vertex.executed_tex_instructions,
            drawlist.vertex.executed_memory_instructions,
        )
        expected_fragment_program = (24, 24, 20, 0, 0)
        actual_fragment_program = (
            drawlist.fragment.program_groups,
            drawlist.fragment.program_instructions,
            drawlist.fragment.program_alu_instructions,
            drawlist.fragment.program_tex_instructions,
            drawlist.fragment.program_memory_instructions,
        )
        expected_fragment_executed = (81920, 0, 0)
        actual_fragment_executed = (
            drawlist.fragment.executed_alu_instructions,
            drawlist.fragment.executed_tex_instructions,
            drawlist.fragment.executed_memory_instructions,
        )
        actual_drawlist = (
            drawlist.drawlist,
            drawlist.draw_id,
            drawlist.vertex.invocations,
            actual_vertex_program,
            actual_vertex_executed,
            drawlist.fragment.invocations,
            actual_fragment_program,
            actual_fragment_executed,
        )
        expected_drawlist = (
            0,
            0,
            25,
            expected_vertex_program,
            expected_vertex_executed,
            4096,
            expected_fragment_program,
            expected_fragment_executed,
        )
        if actual_drawlist != expected_drawlist:
            raise AssertionError(
                f"{options.case}: DrawList {actual_drawlist}, "
                f"expected {expected_drawlist}"
            )

        values = model.values
        relations = (
            (
                values["parameter_coefficient_sets"],
                values["c_primitives"] * 17,
                "seventeen coefficient sets per primitive",
            ),
            (
                values["parameter_write_bytes"],
                values["parameter_coefficient_sets"] * 4 * 4,
                "four dwords per coefficient set",
            ),
            (
                values["pds_douti_issues"],
                values["pds_coefficient_tasks"] * 2,
                "two DOUTI issues per fragment-quad task",
            ),
            (
                values["usc_coefficient_load_bytes"],
                values["pds_coefficient_tasks"] * 68 * 4,
                "sixty-eight coefficient dwords per USC task",
            ),
            (
                values["usc_groups"],
                values["pds_coefficient_tasks"] + 7,
                "fragment tasks plus seven vertex issue groups",
            ),
            (
                values["fragment_candidates"],
                values["ps_invocations"] + values["hsr_rejected_fragments"],
                "HSR candidate conservation",
            ),
            (
                values["pbe_fragment_writes"],
                values["ps_invocations"],
                "one PBE fragment write per executed fragment",
            ),
            (
                values["pixel_data_master_bytes"],
                values["pbe_pixels_written"] * 4,
                "RGBA8 PixelDM payload",
            ),
            (
                values["slc_line_accesses"],
                values["pixel_data_master_bytes"] // 128,
                "128-byte SLC line coverage",
            ),
            (
                values["dram_cycles"],
                values["dram_read_transactions"]
                + values["dram_write_transactions"],
                "one fixed-latency cycle per DRAM transaction",
            ),
            (
                values["virtual_gpu_cycles"],
                values["tiler_cycles"] + values["renderer_cycles"] + 25,
                "frame critical-path composition",
            ),
        )
        for actual, expected, description in relations:
            if actual != expected:
                raise AssertionError(
                    f"{options.case}: {description}: {actual} != {expected}"
                )

    if options.case == "varyings_shader_8":
        expected_standard = {
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
        }
        for field, expected in expected_standard.items():
            golden_value = golden.values.get(field, 0)
            model_value = model.values.get(field, 0)
            if golden_value != expected or model_value != expected:
                raise AssertionError(
                    f"{options.case}: {field} llvmpipe={golden_value}, "
                    f"PvrGPU={model_value}, expected {expected}"
                )

        expected_hello = {
            "mode": "systemc-functional-varyings-shader",
            "functional_scope": "varyings_shader_8-pco-iss-v1",
            "pco_subset": "fmul-fitrp-wdf-fadd-mbyp-uvsw-varying",
            "shader_binary": "mesa-pco-public-encoding",
            "cache_bypass": False,
            "framebuffer_source": "dram-readback",
            "dram_fixed_latency_cycles": 1,
            "tile_width": 32,
            "tile_height": 32,
        }
        for field, expected in expected_hello.items():
            if hello.get(field) != expected:
                raise AssertionError(
                    f"{options.case}: hello {field}={hello.get(field)!r}, "
                    f"expected {expected!r}"
                )

        expected_program_evidence = {
            "vertex_pco_binary": {
                "fingerprint": "fnv1a64:3ea4e650a43484ce",
                "bytes": 176,
            },
            "vertex_pco_opcodes": {
                "fadd": 0,
                "fmul": 2,
                "mbyp": 6,
                "uvsw_write": 9,
                "uvsw_write_emit_endtask": 0,
                "uvsw_emit_endtask": 1,
            },
            "fragment_pco_binary": {
                "fingerprint": "fnv1a64:b1f3b2aa7d58d59d",
                "bytes": 440,
            },
            "fragment_pco_opcodes": {
                "fitrp": 8,
                "wdf": 8,
                "fadd": 28,
                "mbyp": 4,
            },
        }
        for field, expected in expected_program_evidence.items():
            if model_message.get(field) != expected:
                raise AssertionError(
                    f"{options.case}: {field}={model_message.get(field)!r}, "
                    f"expected {expected!r}"
                )

        expected_modeled = {
            "virtual_gpu_cycles": 1392,
            "tiler_cycles": 58,
            "renderer_cycles": 1309,
            "usc_groups": 1159,
            "texture_requests": 0,
            "fifo_stall_events": 0,
            "pool_bytes_in_flight": 0,
            "pool_high_water_bytes": 1347912,
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
        for field, expected in expected_modeled.items():
            actual = model.values.get(field)
            if actual != expected:
                raise AssertionError(
                    f"{options.case}: {field} PvrGPU={actual}, "
                    f"expected {expected}"
                )

        expected_vertex_program = (18, 18, 8, 0, 10)
        actual_vertex_program = (
            drawlist.vertex.program_groups,
            drawlist.vertex.program_instructions,
            drawlist.vertex.program_alu_instructions,
            drawlist.vertex.program_tex_instructions,
            drawlist.vertex.program_memory_instructions,
        )
        expected_vertex_executed = (200, 0, 925)
        actual_vertex_executed = (
            drawlist.vertex.executed_alu_instructions,
            drawlist.vertex.executed_tex_instructions,
            drawlist.vertex.executed_memory_instructions,
        )
        expected_fragment_program = (48, 48, 40, 0, 0)
        actual_fragment_program = (
            drawlist.fragment.program_groups,
            drawlist.fragment.program_instructions,
            drawlist.fragment.program_alu_instructions,
            drawlist.fragment.program_tex_instructions,
            drawlist.fragment.program_memory_instructions,
        )
        expected_fragment_executed = (163840, 0, 0)
        actual_fragment_executed = (
            drawlist.fragment.executed_alu_instructions,
            drawlist.fragment.executed_tex_instructions,
            drawlist.fragment.executed_memory_instructions,
        )
        actual_drawlist = (
            drawlist.drawlist,
            drawlist.draw_id,
            drawlist.vertex.invocations,
            actual_vertex_program,
            actual_vertex_executed,
            drawlist.fragment.invocations,
            actual_fragment_program,
            actual_fragment_executed,
        )
        expected_drawlist = (
            0,
            0,
            25,
            expected_vertex_program,
            expected_vertex_executed,
            4096,
            expected_fragment_program,
            expected_fragment_executed,
        )
        if actual_drawlist != expected_drawlist:
            raise AssertionError(
                f"{options.case}: DrawList {actual_drawlist}, "
                f"expected {expected_drawlist}"
            )

        values = model.values
        relations = (
            (
                values["parameter_coefficient_sets"],
                values["c_primitives"] * 33,
                "thirty-three coefficient sets per primitive",
            ),
            (
                values["parameter_write_bytes"],
                values["parameter_coefficient_sets"] * 4 * 4,
                "four dwords per coefficient set",
            ),
            (
                values["pds_douti_issues"],
                values["pds_coefficient_tasks"] * 2,
                "two DOUTI issues per fragment-quad task",
            ),
            (
                values["usc_coefficient_load_bytes"],
                values["pds_coefficient_tasks"] * 132 * 4,
                "one hundred thirty-two coefficient dwords per USC task",
            ),
            (
                values["usc_groups"],
                values["pds_coefficient_tasks"] + 7,
                "fragment tasks plus seven vertex issue groups",
            ),
            (
                values["fragment_candidates"],
                values["ps_invocations"] + values["hsr_rejected_fragments"],
                "HSR candidate conservation",
            ),
            (
                values["pbe_fragment_writes"],
                values["ps_invocations"],
                "one PBE fragment write per executed fragment",
            ),
            (
                values["pixel_data_master_bytes"],
                values["pbe_pixels_written"] * 4,
                "RGBA8 PixelDM payload",
            ),
            (
                values["slc_line_accesses"],
                values["pixel_data_master_bytes"] // 128,
                "128-byte SLC line coverage",
            ),
            (
                values["dram_cycles"],
                values["dram_read_transactions"]
                + values["dram_write_transactions"],
                "one fixed-latency cycle per DRAM transaction",
            ),
            (
                values["virtual_gpu_cycles"],
                values["tiler_cycles"] + values["renderer_cycles"] + 25,
                "frame critical-path composition",
            ),
        )
        for actual, expected, description in relations:
            if actual != expected:
                raise AssertionError(
                    f"{options.case}: {description}: {actual} != {expected}"
                )

    if options.case == "fill_tex_nearest":
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
            golden_value = golden.values.get(field, 0)
            model_value = model.values.get(field, 0)
            if golden_value != expected or model_value != expected:
                raise AssertionError(
                    f"{options.case}: {field} llvmpipe={golden_value}, "
                    f"PvrGPU={model_value}, expected {expected}"
                )

        expected_hello = {
            "backend": "pvrgpu",
            "mode": "systemc-functional-fill-texture-nearest",
            "functional_scope": "fill_tex_nearest-pco-iss-v1",
            "command_source": "builtin-glbench-fixture",
            "mesa_command_ingest": False,
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
            if hello.get(field) != expected:
                raise AssertionError(
                    f"{options.case}: hello {field}={hello.get(field)!r}, "
                    f"expected {expected!r}"
                )
        expected_caches = {
            "tcu_cache": {
                "capacity_bytes": 24 * 1024,
                "line_bytes": 64,
                "ways": 4,
                "banks": 4,
            },
            "slc_cache": {
                "capacity_bytes": 2 * 1024 * 1024,
                "line_bytes": 128,
                "ways": 8,
                "banks": 8,
            },
        }
        for field, expected in expected_caches.items():
            if hello.get(field) != expected:
                raise AssertionError(
                    f"{options.case}: hello {field}={hello.get(field)!r}, "
                    f"expected {expected!r}"
                )
        expected_counter_provenance = {
            "backend": "pvrgpu",
            "source": "pvrgpu-systemc",
            "provenance": "modeled",
            "functional_scope": "fill_tex_nearest-pco-iss-v1",
            "command_source": "builtin-glbench-fixture",
            "timing_provenance": "uncalibrated",
            "cache_bypass": False,
            "framebuffer_source": "dram-readback",
        }
        for field, expected in expected_counter_provenance.items():
            if model_message.get(field) != expected:
                raise AssertionError(
                    f"{options.case}: counter {field}="
                    f"{model_message.get(field)!r}, expected {expected!r}"
                )

        expected_program_evidence = {
            "vertex_pco_binary": {
                "fingerprint": "fnv1a64:36c31424e4119557",
                "bytes": 80,
            },
            "vertex_pco_opcodes": {
                "fadd": 0,
                "fmul": 2,
                "mbyp": 3,
                "uvsw_write": 2,
                "uvsw_write_emit_endtask": 0,
                "uvsw_emit_endtask": 1,
            },
            "fragment_pco_binary": {
                "fingerprint": "fnv1a64:0693891931816150",
                "bytes": 184,
            },
            "fragment_pco_opcodes": {
                "fitrp": 1,
                "wdf": 2,
                "smp": 1,
                "mbyp": 18,
            },
        }
        for field, expected in expected_program_evidence.items():
            if model_message.get(field) != expected:
                raise AssertionError(
                    f"{options.case}: {field}={model_message.get(field)!r}, "
                    f"expected {expected!r}"
                )

        values = model.values
        expected_modeled = {
            "virtual_gpu_cycles": 17459,
            "tiler_cycles": 51,
            "renderer_cycles": 17383,
            "usc_groups": 1089,
            "texture_requests": 4352,
            "fifo_stall_events": 0,
            "pool_bytes_in_flight": 0,
            # Includes quad_id/quad_lane in the TextureSampleRequest ABI.
            "pool_high_water_bytes": 3246322,
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
            actual = values.get(field)
            if actual != expected:
                raise AssertionError(
                    f"{options.case}: {field} PvrGPU={actual}, "
                    f"expected {expected}"
                )

        expected_vertex_program = (8, 8, 5, 0, 3)
        actual_vertex_program = (
            drawlist.vertex.program_groups,
            drawlist.vertex.program_instructions,
            drawlist.vertex.program_alu_instructions,
            drawlist.vertex.program_tex_instructions,
            drawlist.vertex.program_memory_instructions,
        )
        expected_vertex_executed = (20, 0, 28)
        actual_vertex_executed = (
            drawlist.vertex.executed_alu_instructions,
            drawlist.vertex.executed_tex_instructions,
            drawlist.vertex.executed_memory_instructions,
        )
        expected_fragment_program = (22, 22, 19, 1, 0)
        actual_fragment_program = (
            drawlist.fragment.program_groups,
            drawlist.fragment.program_instructions,
            drawlist.fragment.program_alu_instructions,
            drawlist.fragment.program_tex_instructions,
            drawlist.fragment.program_memory_instructions,
        )
        expected_fragment_executed = (82688, 4352, 0)
        actual_fragment_executed = (
            drawlist.fragment.executed_alu_instructions,
            drawlist.fragment.executed_tex_instructions,
            drawlist.fragment.executed_memory_instructions,
        )
        actual_drawlist = (
            drawlist.drawlist,
            drawlist.draw_id,
            drawlist.vertex.invocations,
            actual_vertex_program,
            actual_vertex_executed,
            drawlist.fragment.invocations,
            actual_fragment_program,
            actual_fragment_executed,
        )
        expected_drawlist = (
            0,
            0,
            4,
            expected_vertex_program,
            expected_vertex_executed,
            4096,
            expected_fragment_program,
            expected_fragment_executed,
        )
        if actual_drawlist != expected_drawlist:
            raise AssertionError(
                f"{options.case}: DrawList {actual_drawlist}, "
                f"expected {expected_drawlist}"
            )

        relations = (
            (
                values["parameter_coefficient_sets"],
                values["c_primitives"] * 3,
                "three coefficient sets per primitive",
            ),
            (
                values["parameter_write_bytes"],
                values["parameter_coefficient_sets"] * 4 * 4,
                "four dwords per coefficient set",
            ),
            (
                values["pds_douti_issues"],
                values["pds_coefficient_tasks"] * 2,
                "two DOUTI issues per fragment-quad task",
            ),
            (
                values["usc_coefficient_load_bytes"],
                values["pds_coefficient_tasks"] * 12 * 4,
                "twelve coefficient dwords per USC task",
            ),
            (
                values["usc_groups"],
                values["pds_coefficient_tasks"] + 1,
                "fragment tasks plus one vertex issue group",
            ),
            (
                values["texture_requests"],
                values["texel_fetches"],
                "one texture request per sampled lane",
            ),
            (
                values["fs_tex_instructions"],
                values["texture_requests"],
                "one executed SMP per texture request",
            ),
            (
                values["fs_alu_instructions"],
                values["texture_requests"]
                * drawlist.fragment.program_alu_instructions,
                "fragment ALU groups per sampled lane",
            ),
            (
                values["texture_cycles"],
                values["texture_requests"] * 2,
                "two texture-processing cycles per request",
            ),
            (
                values["tcu_line_accesses"],
                values["texture_requests"],
                "one TCU line lookup per texture request",
            ),
            (
                values["tcu_hits"] + values["tcu_misses"],
                values["tcu_line_accesses"],
                "TCU hit/miss conservation",
            ),
            (
                values["tcu_cycles"],
                values["tcu_line_accesses"],
                "one TCU cycle per line access",
            ),
            (
                values["slc_read_accesses"],
                values["tcu_misses"],
                "one SLC read per TCU miss",
            ),
            (
                values["slc_line_accesses"],
                values["slc_read_accesses"] + values["slc_write_accesses"],
                "SLC read/write conservation",
            ),
            (
                values["slc_hits"] + values["slc_misses"],
                values["slc_line_accesses"],
                "SLC hit/miss conservation",
            ),
            (
                values["dram_read_transactions"],
                values["slc_misses"] - values["slc_write_accesses"] + 1,
                "texture SLC read misses plus framebuffer readback",
            ),
            (
                values["dram_read_bytes"],
                (values["dram_read_transactions"] - 1) * 128
                + values["framebuffer_dram_readback_bytes"],
                "texture lines plus framebuffer readback bytes",
            ),
            (
                values["dram_write_transactions"],
                values["slc_writebacks"],
                "one DRAM write per SLC writeback",
            ),
            (
                values["dram_write_bytes"],
                values["dram_write_transactions"] * 128,
                "one 128-byte line per DRAM write",
            ),
            (
                values["dram_cycles"],
                values["dram_read_transactions"]
                + values["dram_write_transactions"],
                "one fixed-latency cycle per DRAM transaction",
            ),
            (
                values["fragment_candidates"],
                values["ps_invocations"] + values["hsr_rejected_fragments"],
                "HSR candidate conservation",
            ),
            (
                values["pbe_fragment_writes"],
                values["ps_invocations"],
                "one PBE fragment write per executed fragment",
            ),
            (
                values["pixel_data_master_bytes"],
                values["pbe_pixels_written"] * 4,
                "RGBA8 PixelDM payload",
            ),
            (
                values["virtual_gpu_cycles"],
                values["tiler_cycles"] + values["renderer_cycles"] + 25,
                "frame critical-path composition",
            ),
        )
        for actual, expected, description in relations:
            if actual != expected:
                raise AssertionError(
                    f"{options.case}: {description}: {actual} != {expected}"
                )

        artifact_value = model_message.get("artifact_png")
        if not isinstance(artifact_value, str):
            raise AssertionError("fill_tex_nearest: missing PNG artifact path")
        artifact = Path(artifact_value)
        if artifact.name != "fill_tex_nearest_sample_000001.png" or (
            not artifact.is_file()
        ):
            raise AssertionError("fill_tex_nearest: invalid PNG artifact")

    if options.case == "fill_tex_bilinear":
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
            golden_value = golden.values.get(field, 0)
            model_value = model.values.get(field, 0)
            if golden_value != expected or model_value != expected:
                raise AssertionError(
                    f"{options.case}: {field} llvmpipe={golden_value}, "
                    f"PvrGPU={model_value}, expected {expected}"
                )

        expected_hello = {
            "backend": "pvrgpu",
            "mode": "systemc-functional-fill-texture-bilinear",
            "functional_scope": "fill_tex_bilinear-pco-iss-v1",
            "command_source": "builtin-glbench-fixture",
            "mesa_command_ingest": False,
            "shader_binary": "mesa-pco-public-encoding",
            "pco_subset": "fmul-fitrp-wdf-smp-mbyp-uvsw-texture",
            "workload": "fill_tex_bilinear",
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
            if hello.get(field) != expected:
                raise AssertionError(
                    f"{options.case}: hello {field}={hello.get(field)!r}, "
                    f"expected {expected!r}"
                )
        expected_caches = {
            "tcu_cache": {
                "capacity_bytes": 24 * 1024,
                "line_bytes": 64,
                "ways": 4,
                "banks": 4,
            },
            "slc_cache": {
                "capacity_bytes": 2 * 1024 * 1024,
                "line_bytes": 128,
                "ways": 8,
                "banks": 8,
            },
        }
        for field, expected in expected_caches.items():
            if hello.get(field) != expected:
                raise AssertionError(
                    f"{options.case}: hello {field}={hello.get(field)!r}, "
                    f"expected {expected!r}"
                )

        expected_counter_provenance = {
            "backend": "pvrgpu",
            "source": "pvrgpu-systemc",
            "provenance": "modeled",
            "functional_scope": "fill_tex_bilinear-pco-iss-v1",
            "command_source": "builtin-glbench-fixture",
            "timing_provenance": "uncalibrated",
            "cache_bypass": False,
            "framebuffer_source": "dram-readback",
            "marker": "fill_tex_bilinear",
        }
        for field, expected in expected_counter_provenance.items():
            if model_message.get(field) != expected:
                raise AssertionError(
                    f"{options.case}: counter {field}="
                    f"{model_message.get(field)!r}, expected {expected!r}"
                )

        expected_program_evidence = {
            "vertex_pco_binary": {
                "fingerprint": "fnv1a64:36c31424e4119557",
                "bytes": 80,
            },
            "vertex_pco_opcodes": {
                "fadd": 0,
                "fmul": 2,
                "mbyp": 3,
                "uvsw_write": 2,
                "uvsw_write_emit_endtask": 0,
                "uvsw_emit_endtask": 1,
            },
            "fragment_pco_binary": {
                "fingerprint": "fnv1a64:0693891931816150",
                "bytes": 184,
            },
            "fragment_pco_opcodes": {
                "fitrp": 1,
                "wdf": 2,
                "smp": 1,
                "mbyp": 18,
            },
        }
        for field, expected in expected_program_evidence.items():
            if model_message.get(field) != expected:
                raise AssertionError(
                    f"{options.case}: {field}={model_message.get(field)!r}, "
                    f"expected {expected!r}"
                )

        values = model.values
        expected_modeled = {
            "virtual_gpu_cycles": 33587,
            "tiler_cycles": 51,
            "renderer_cycles": 33511,
            "usc_groups": 1089,
            "texture_requests": 4352,
            "fifo_stall_events": 0,
            "pool_bytes_in_flight": 0,
            # Includes quad_id/quad_lane in the TextureSampleRequest ABI.
            "pool_high_water_bytes": 3246322,
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
            actual = values.get(field)
            if actual != expected:
                raise AssertionError(
                    f"{options.case}: {field} PvrGPU={actual}, "
                    f"expected {expected}"
                )
        expected_vertex_program = (8, 8, 5, 0, 3)
        actual_vertex_program = (
            drawlist.vertex.program_groups,
            drawlist.vertex.program_instructions,
            drawlist.vertex.program_alu_instructions,
            drawlist.vertex.program_tex_instructions,
            drawlist.vertex.program_memory_instructions,
        )
        expected_vertex_executed = (20, 0, 28)
        actual_vertex_executed = (
            drawlist.vertex.executed_alu_instructions,
            drawlist.vertex.executed_tex_instructions,
            drawlist.vertex.executed_memory_instructions,
        )
        expected_fragment_program = (22, 22, 19, 1, 0)
        actual_fragment_program = (
            drawlist.fragment.program_groups,
            drawlist.fragment.program_instructions,
            drawlist.fragment.program_alu_instructions,
            drawlist.fragment.program_tex_instructions,
            drawlist.fragment.program_memory_instructions,
        )
        expected_fragment_executed = (82688, 4352, 0)
        actual_fragment_executed = (
            drawlist.fragment.executed_alu_instructions,
            drawlist.fragment.executed_tex_instructions,
            drawlist.fragment.executed_memory_instructions,
        )
        actual_drawlist = (
            drawlist.drawlist,
            drawlist.draw_id,
            drawlist.vertex.invocations,
            actual_vertex_program,
            actual_vertex_executed,
            drawlist.fragment.invocations,
            actual_fragment_program,
            actual_fragment_executed,
        )
        expected_drawlist = (
            0,
            0,
            4,
            expected_vertex_program,
            expected_vertex_executed,
            4096,
            expected_fragment_program,
            expected_fragment_executed,
        )
        if actual_drawlist != expected_drawlist:
            raise AssertionError(
                f"{options.case}: DrawList {actual_drawlist}, "
                f"expected {expected_drawlist}"
            )

        relations = (
            (
                values["parameter_coefficient_sets"],
                values["c_primitives"] * 3,
                "three coefficient sets per primitive",
            ),
            (
                values["parameter_write_bytes"],
                values["parameter_coefficient_sets"] * 4 * 4,
                "four dwords per coefficient set",
            ),
            (
                values["pds_douti_issues"],
                values["pds_coefficient_tasks"] * 2,
                "two DOUTI issues per fragment-quad task",
            ),
            (
                values["usc_coefficient_load_bytes"],
                values["pds_coefficient_tasks"] * 12 * 4,
                "twelve coefficient dwords per USC task",
            ),
            (
                values["usc_groups"],
                values["pds_coefficient_tasks"] + 1,
                "fragment tasks plus one vertex issue group",
            ),
            (
                values["texture_requests"],
                values["fs_tex_instructions"],
                "one SMP request per executed texture instruction",
            ),
            (
                values["texel_fetches"],
                values["texture_requests"] * 4,
                "four bilinear taps per logical texture request",
            ),
            (
                values["fs_alu_instructions"],
                values["texture_requests"]
                * drawlist.fragment.program_alu_instructions,
                "fragment ALU groups per sampled lane",
            ),
            (
                values["texture_cycles"],
                values["texture_requests"] * 2,
                "two texture-processing cycles per logical SMP request",
            ),
            (
                values["tcu_line_accesses"],
                values["texel_fetches"],
                "one TCU line lookup per bilinear tap",
            ),
            (
                values["tcu_read_accesses"],
                values["tcu_line_accesses"],
                "read-only TCU access conservation",
            ),
            (
                values["tcu_hits"] + values["tcu_misses"],
                values["tcu_line_accesses"],
                "TCU hit/miss conservation",
            ),
            (
                values["tcu_cycles"],
                values["tcu_line_accesses"],
                "one TCU cycle per line access",
            ),
            (
                values["slc_read_accesses"],
                values["tcu_misses"],
                "one SLC read per TCU miss",
            ),
            (
                values["slc_line_accesses"],
                values["slc_read_accesses"] + values["slc_write_accesses"],
                "SLC read/write conservation",
            ),
            (
                values["slc_hits"] + values["slc_misses"],
                values["slc_line_accesses"],
                "SLC hit/miss conservation",
            ),
            (
                values["dram_read_transactions"],
                values["slc_misses"] - values["slc_write_accesses"] + 1,
                "texture SLC read misses plus framebuffer readback",
            ),
            (
                values["dram_read_bytes"],
                (values["dram_read_transactions"] - 1) * 128
                + values["framebuffer_dram_readback_bytes"],
                "texture lines plus framebuffer readback bytes",
            ),
            (
                values["dram_write_transactions"],
                values["slc_writebacks"],
                "one DRAM write per SLC writeback",
            ),
            (
                values["dram_write_bytes"],
                values["dram_write_transactions"] * 128,
                "one 128-byte line per DRAM write",
            ),
            (
                values["dram_cycles"],
                values["dram_read_transactions"]
                + values["dram_write_transactions"],
                "one fixed-latency cycle per DRAM transaction",
            ),
            (
                values["fragment_candidates"],
                values["ps_invocations"] + values["hsr_rejected_fragments"],
                "HSR candidate conservation",
            ),
            (
                values["pbe_fragment_writes"],
                values["ps_invocations"],
                "one PBE fragment write per executed fragment",
            ),
            (
                values["pixel_data_master_bytes"],
                values["pbe_pixels_written"] * 4,
                "RGBA8 PixelDM payload",
            ),
            (
                values["virtual_gpu_cycles"],
                values["tiler_cycles"] + values["renderer_cycles"] + 25,
                "frame critical-path composition",
            ),
        )
        for actual, expected, description in relations:
            if actual != expected:
                raise AssertionError(
                    f"{options.case}: {description}: {actual} != {expected}"
                )

        artifact_value = model_message.get("artifact_png")
        if not isinstance(artifact_value, str):
            raise AssertionError("fill_tex_bilinear: missing PNG artifact path")
        artifact = Path(artifact_value)
        if artifact.name != "fill_tex_bilinear_sample_000001.png" or (
            not artifact.is_file()
        ):
            raise AssertionError("fill_tex_bilinear: invalid PNG artifact")

    trilinear_profiles = {
        "fill_tex_trilinear_linear_01": {
            "mode": "systemc-functional-fill-texture-trilinear-linear-01",
            "ps_invocations": 3600,
            "texel_fetches": 32640,
            "virtual_gpu_cycles": 42551,
            "renderer_cycles": 42475,
            "usc_groups": 1021,
            "texture_requests": 4080,
            "pool_high_water_bytes": 3108130,
            "fs_alu_instructions": 77520,
            "usc_slot_cycles": 515,
            "usc_cluster_cycles": 264,
            "pds_coefficient_tasks": 1020,
            "pds_douti_issues": 2040,
            "usc_coefficient_load_bytes": 48960,
            "isp_cycles": 67,
            "fragment_frontend_cycles": 37,
            "texture_cycles": 8160,
            "tcu_hits": 32320,
        },
        "fill_tex_trilinear_linear_04": {
            "mode": "systemc-functional-fill-texture-trilinear-linear-04",
            "ps_invocations": 2304,
            "texel_fetches": 19968,
            "virtual_gpu_cycles": 26381,
            "renderer_cycles": 26305,
            "usc_groups": 625,
            "texture_requests": 2496,
            "pool_high_water_bytes": 2456242,
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
        },
        "fill_tex_trilinear_linear_05": {
            "mode": "systemc-functional-fill-texture-trilinear-linear-05",
            "ps_invocations": 2116,
            "texel_fetches": 19904,
            "virtual_gpu_cycles": 26298,
            "renderer_cycles": 26222,
            "usc_groups": 623,
            "texture_requests": 2488,
            "pool_high_water_bytes": 2435530,
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
        },
    }
    if options.case in trilinear_profiles:
        profile = trilinear_profiles[options.case]
        trilinear_scope = f"{options.case}-pco-iss-v1"
        expected_standard = {
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
        }
        for field, expected in expected_standard.items():
            golden_value = golden.values.get(field, 0)
            model_value = model.values.get(field, 0)
            if golden_value != expected or model_value != expected:
                raise AssertionError(
                    f"{options.case}: {field} llvmpipe={golden_value}, "
                    f"PvrGPU={model_value}, expected {expected}"
                )

        expected_hello = {
            "backend": "pvrgpu",
            "mode": profile["mode"],
            "functional_scope": trilinear_scope,
            "command_source": "builtin-glbench-fixture",
            "mesa_command_ingest": False,
            "shader_binary": "mesa-pco-public-encoding",
            "pco_subset": "fmul-fitrp-wdf-smp-mbyp-uvsw-texture",
            "workload": options.case,
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
            if hello.get(field) != expected:
                raise AssertionError(
                    f"{options.case}: hello {field}={hello.get(field)!r}, "
                    f"expected {expected!r}"
                )
        expected_caches = {
            "tcu_cache": {
                "capacity_bytes": 24 * 1024,
                "line_bytes": 64,
                "ways": 4,
                "banks": 4,
            },
            "slc_cache": {
                "capacity_bytes": 2 * 1024 * 1024,
                "line_bytes": 128,
                "ways": 8,
                "banks": 8,
            },
        }
        for field, expected in expected_caches.items():
            if hello.get(field) != expected:
                raise AssertionError(
                    f"{options.case}: hello {field}={hello.get(field)!r}, "
                    f"expected {expected!r}"
                )

        expected_counter_provenance = {
            "backend": "pvrgpu",
            "source": "pvrgpu-systemc",
            "provenance": "modeled",
            "functional_scope": trilinear_scope,
            "command_source": "builtin-glbench-fixture",
            "timing_provenance": "uncalibrated",
            "cache_bypass": False,
            "framebuffer_source": "dram-readback",
            "marker": options.case,
        }
        for field, expected in expected_counter_provenance.items():
            if model_message.get(field) != expected:
                raise AssertionError(
                    f"{options.case}: counter {field}="
                    f"{model_message.get(field)!r}, expected {expected!r}"
                )

        expected_program_evidence = {
            "vertex_pco_binary": {
                "fingerprint": "fnv1a64:36c31424e4119557",
                "bytes": 80,
            },
            "vertex_pco_opcodes": {
                "fadd": 0,
                "fmul": 2,
                "mbyp": 3,
                "uvsw_write": 2,
                "uvsw_write_emit_endtask": 0,
                "uvsw_emit_endtask": 1,
            },
            "fragment_pco_binary": {
                "fingerprint": "fnv1a64:0693891931816150",
                "bytes": 184,
            },
            "fragment_pco_opcodes": {
                "fitrp": 1,
                "wdf": 2,
                "smp": 1,
                "mbyp": 18,
            },
        }
        for field, expected in expected_program_evidence.items():
            if model_message.get(field) != expected:
                raise AssertionError(
                    f"{options.case}: {field}={model_message.get(field)!r}, "
                    f"expected {expected!r}"
                )

        values = model.values
        expected_modeled = {
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
            "usc_coefficient_load_bytes": profile[
                "usc_coefficient_load_bytes"
            ],
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
        for field, expected in expected_modeled.items():
            actual = values.get(field)
            if actual != expected:
                raise AssertionError(
                    f"{options.case}: {field} PvrGPU={actual}, "
                    f"expected {expected}"
                )
        expected_counter_fields = set(expected_standard) | set(expected_modeled)
        if set(values) != expected_counter_fields:
            raise AssertionError(
                f"{options.case}: counter fields={sorted(values)}, "
                f"expected {sorted(expected_counter_fields)}"
            )

        expected_vertex_program = (8, 8, 5, 0, 3)
        actual_vertex_program = (
            drawlist.vertex.program_groups,
            drawlist.vertex.program_instructions,
            drawlist.vertex.program_alu_instructions,
            drawlist.vertex.program_tex_instructions,
            drawlist.vertex.program_memory_instructions,
        )
        expected_vertex_executed = (20, 0, 28)
        actual_vertex_executed = (
            drawlist.vertex.executed_alu_instructions,
            drawlist.vertex.executed_tex_instructions,
            drawlist.vertex.executed_memory_instructions,
        )
        expected_fragment_program = (22, 22, 19, 1, 0)
        actual_fragment_program = (
            drawlist.fragment.program_groups,
            drawlist.fragment.program_instructions,
            drawlist.fragment.program_alu_instructions,
            drawlist.fragment.program_tex_instructions,
            drawlist.fragment.program_memory_instructions,
        )
        expected_fragment_executed = (
            profile["fs_alu_instructions"],
            profile["texture_requests"],
            0,
        )
        actual_fragment_executed = (
            drawlist.fragment.executed_alu_instructions,
            drawlist.fragment.executed_tex_instructions,
            drawlist.fragment.executed_memory_instructions,
        )
        actual_drawlist = (
            drawlist.drawlist,
            drawlist.draw_id,
            drawlist.vertex.invocations,
            actual_vertex_program,
            actual_vertex_executed,
            drawlist.fragment.invocations,
            actual_fragment_program,
            actual_fragment_executed,
        )
        expected_drawlist = (
            0,
            0,
            4,
            expected_vertex_program,
            expected_vertex_executed,
            profile["ps_invocations"],
            expected_fragment_program,
            expected_fragment_executed,
        )
        if actual_drawlist != expected_drawlist:
            raise AssertionError(
                f"{options.case}: DrawList {actual_drawlist}, "
                f"expected {expected_drawlist}"
            )

        relations = (
            (
                values["parameter_coefficient_sets"],
                values["c_primitives"] * 3,
                "three coefficient sets per primitive",
            ),
            (
                values["parameter_write_bytes"],
                values["parameter_coefficient_sets"] * 4 * 4,
                "four dwords per coefficient set",
            ),
            (
                values["pds_douti_issues"],
                values["pds_coefficient_tasks"] * 2,
                "two DOUTI issues per fragment-quad task",
            ),
            (
                values["usc_coefficient_load_bytes"],
                values["pds_coefficient_tasks"] * 12 * 4,
                "twelve coefficient dwords per USC task",
            ),
            (
                values["usc_groups"],
                values["pds_coefficient_tasks"] + 1,
                "fragment tasks plus one vertex issue group",
            ),
            (
                values["texture_requests"],
                values["pds_coefficient_tasks"] * 4,
                "four texture lanes per fragment-quad task",
            ),
            (
                values["texture_requests"],
                values["fs_tex_instructions"],
                "one SMP request per executed texture instruction",
            ),
            (
                values["texel_fetches"],
                values["texture_requests"] * 8,
                "eight bilinear mip taps per trilinear texture request",
            ),
            (
                values["fs_alu_instructions"],
                values["texture_requests"]
                * drawlist.fragment.program_alu_instructions,
                "fragment ALU groups per sampled lane",
            ),
            (
                values["texture_cycles"],
                values["texture_requests"] * 2,
                "two texture-processing cycles per logical SMP request",
            ),
            (
                values["tcu_line_accesses"],
                values["texel_fetches"],
                "one TCU line lookup per trilinear tap",
            ),
            (
                values["tcu_read_accesses"],
                values["tcu_line_accesses"],
                "read-only TCU access conservation",
            ),
            (
                values["tcu_hits"] + values["tcu_misses"],
                values["tcu_line_accesses"],
                "TCU hit/miss conservation",
            ),
            (
                values["tcu_cycles"],
                values["tcu_line_accesses"],
                "one TCU cycle per line access",
            ),
            (
                values["slc_read_accesses"],
                values["tcu_misses"],
                "one SLC read per TCU miss",
            ),
            (
                values["slc_write_accesses"],
                values["pixel_data_master_bytes"] // 128,
                "one SLC write per RGBA8 framebuffer line",
            ),
            (
                values["slc_line_accesses"],
                values["slc_read_accesses"] + values["slc_write_accesses"],
                "SLC read/write conservation",
            ),
            (
                values["slc_hits"] + values["slc_misses"],
                values["slc_line_accesses"],
                "SLC hit/miss conservation",
            ),
            (
                values["slc_writebacks"],
                values["slc_write_accesses"],
                "framebuffer SLC writes are flushed to DRAM",
            ),
            (
                values["dram_read_transactions"],
                values["slc_misses"] - values["slc_write_accesses"] + 1,
                "texture SLC read misses plus framebuffer readback",
            ),
            (
                values["dram_read_bytes"],
                (values["dram_read_transactions"] - 1) * 128
                + values["framebuffer_dram_readback_bytes"],
                "texture lines plus framebuffer readback bytes",
            ),
            (
                values["dram_write_transactions"],
                values["slc_writebacks"],
                "one DRAM write per SLC writeback",
            ),
            (
                values["dram_write_bytes"],
                values["dram_write_transactions"] * 128,
                "one 128-byte line per DRAM write",
            ),
            (
                values["dram_cycles"],
                values["dram_read_transactions"]
                + values["dram_write_transactions"],
                "one fixed-latency cycle per DRAM transaction",
            ),
            (
                values["fragment_candidates"],
                values["ps_invocations"] + values["hsr_rejected_fragments"],
                "HSR candidate conservation",
            ),
            (
                values["pbe_fragment_writes"],
                values["ps_invocations"],
                "one PBE fragment write per executed fragment",
            ),
            (
                values["pixel_data_master_bytes"],
                values["pbe_pixels_written"] * 4,
                "RGBA8 PixelDM payload",
            ),
            (
                values["virtual_gpu_cycles"],
                values["tiler_cycles"] + values["renderer_cycles"] + 25,
                "frame critical-path composition",
            ),
        )
        for actual, expected, description in relations:
            if actual != expected:
                raise AssertionError(
                    f"{options.case}: {description}: {actual} != {expected}"
                )

        artifact_value = model_message.get("artifact_png")
        if not isinstance(artifact_value, str):
            raise AssertionError(f"{options.case}: missing PNG artifact path")
        artifact = Path(artifact_value)
        expected_artifact_name = f"{options.case}_sample_000001.png"
        if artifact.name != expected_artifact_name or not artifact.is_file():
            raise AssertionError(f"{options.case}: invalid PNG artifact")

    if options.case == "fill_solid_blended":
        expected_blend = {
            "enabled": True,
            "rgb_equation": "add",
            "alpha_equation": "add",
            "source_rgb_factor": "source-alpha",
            "destination_rgb_factor": "one-minus-source-alpha",
            "source_alpha_factor": "source-alpha",
            "destination_alpha_factor": "one-minus-source-alpha",
        }
        if hello.get("blend_state") != expected_blend:
            raise AssertionError("PvrGPU hello blend state does not match GLBench")
        if hello.get("effective_early_hsr") is not False:
            raise AssertionError("Blending incorrectly reports effective early HSR")
        for field in (
            "pbe_color_reads",
            "pbe_blended_fragments",
            "pbe_fragment_writes",
        ):
            if model.values.get(field) != ps_invocations:
                raise AssertionError(
                    f"{field}={model.values.get(field)!r}, expected {ps_invocations}"
                )
        if model.values.get("hsr_rejected_fragments") != 0:
            raise AssertionError("Blend path discarded a passing fragment in HSR")
    else:
        if model.values.get("pbe_color_reads") != 0 or model.values.get(
            "pbe_blended_fragments"
        ) != 0:
            raise AssertionError("Non-blended case performed PBE blend reads/ops")
        if model.values.get("pbe_fragment_writes") != ps_invocations:
            raise AssertionError("PBE fragment writes do not match PS invocations")

    print(
        f"COUNTER_MATCH case={options.case} fields={len(PIPELINE_FIELDS)} "
        f"ps_invocations={ps_invocations} drawlists=1 pool_leaks=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
