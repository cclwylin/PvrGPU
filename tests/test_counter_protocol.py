from __future__ import annotations

import json
import copy
from pathlib import Path
import tempfile
import unittest

from counter_protocol import (
    ALL_COUNTER_FIELDS,
    COUNTER_INFO,
    MODEL_COUNTER_FIELDS,
    CounterProtocolError,
    counter_record_from_message,
    parse_jsonl_line,
    parse_markdown_report,
)


class CounterProtocolTests(unittest.TestCase):
    def test_tessellation_catalog_preserves_patch_lane_and_byte_units(self) -> None:
        expected_units = {
            "tcs_invocations": "invocations",
            **{f"tcs_{kind}_instructions": "instructions" for kind in ("alu", "memory", "load", "store")},
            **{f"tcs_{scope}_{direction}_bytes": "bytes" for scope in ("input", "output") for direction in ("read", "write")},
            **{f"tes_{kind}_instructions": "instructions" for kind in ("alu", "memory", "load")},
            "tes_patch_read_bytes": "bytes", "tessellation_patches": "patches",
            "tessellation_primitives": "primitives", "tessellation_domain_write_bytes": "bytes",
            "tessellation_domain_read_bytes": "bytes", "tessellation_level_read_bytes": "bytes",
        }
        self.assertEqual(len(expected_units), 18)
        self.assertTrue(expected_units.keys() <= set(MODEL_COUNTER_FIELDS))
        for field, unit in expected_units.items():
            self.assertEqual(COUNTER_INFO[field][1], unit)
        self.assertEqual(COUNTER_INFO["hs_invocations"][1], "patches")
        self.assertEqual(COUNTER_INFO["ds_invocations"][1], "invocations")

    def test_tessellation_drawlists_preserve_actual_stage_work(self) -> None:
        empty = {"invocations": 0, "program": {"groups": 0, "instructions": 0,
                 "alu": 0, "tex": 0, "memory": 0},
                 "executed": {"alu": 0, "tex": 0, "memory": 0}}
        tcs = {"invocations": 4, "program": {"groups": 7, "instructions": 9,
               "alu": 3, "tex": 0, "memory": 4},
               "executed": {"alu": 12, "tex": 0, "memory": 8}}
        tes = {"invocations": 9, "program": {"groups": 8, "instructions": 10,
               "alu": 3, "tex": 0, "memory": 5},
               "executed": {"alu": 27, "tex": 0, "memory": 45}}
        counters = {"drawlists": 2, "hs_invocations": 2, "tcs_invocations": 8,
                    "ds_invocations": 9, "tcs_alu_instructions": 24,
                    "tcs_memory_instructions": 16, "tes_alu_instructions": 27,
                    "tes_memory_instructions": 45, "tessellation_patches": 2,
                    "tessellation_level_read_bytes": 48,
                    "tessellation_domain_write_bytes": 72,
                    "tessellation_domain_read_bytes": 72}
        culled_tes = copy.deepcopy(tes)
        culled_tes["invocations"] = 0
        culled_tes["executed"] = {"alu": 0, "tex": 0, "memory": 0}
        message = {"schema": "pvrgpu.counter.v1", "type": "counter", "counters": counters,
                   "drawlist_stats": [
                       {"drawlist": 0, "draw_id": 11, "vs": empty, "fs": empty,
                        "tcs": tcs, "tes": tes},
                       {"drawlist": 1, "draw_id": 12, "vs": empty, "fs": empty,
                        "tcs": tcs, "tes": culled_tes}]}
        record = counter_record_from_message(message)
        self.assertEqual(record.values["hs_invocations"], 2)
        self.assertEqual(record.drawlist_stats[0].tessellation_control.invocations, 4)
        self.assertEqual(record.drawlist_stats[0].tessellation_evaluation.invocations, 9)
        self.assertEqual(record.drawlist_stats[1].tessellation_evaluation.invocations, 0)
        self.assertGreater(record.drawlist_stats[1].tessellation_evaluation.program_instructions, 0)
        for field in ("tcs_invocations", "ds_invocations", "tcs_alu_instructions",
                      "tcs_memory_instructions", "tes_alu_instructions", "tes_memory_instructions",
                      "hs_invocations", "tessellation_patches"):
            changed = copy.deepcopy(message)
            changed["counters"][field] += 1
            with self.subTest(counter=field), self.assertRaises(CounterProtocolError):
                counter_record_from_message(changed)
        for missing in ("tcs", "tes"):
            changed = copy.deepcopy(message)
            del changed["drawlist_stats"][0][missing]
            with self.subTest(missing=missing), self.assertRaises(CounterProtocolError):
                counter_record_from_message(changed)
        for stage in ("tcs", "tes"):
            changed = copy.deepcopy(message)
            changed["drawlist_stats"][0][stage]["executed"]["tex"] = 1
            with self.subTest(stage=stage), self.assertRaises(CounterProtocolError):
                counter_record_from_message(changed)

    def test_unsubmitted_tessellation_stays_absent(self) -> None:
        empty = {"invocations": 0, "program": {"groups": 0, "instructions": 0,
                 "alu": 0, "tex": 0, "memory": 0},
                 "executed": {"alu": 0, "tex": 0, "memory": 0}}
        message = {"schema": "pvrgpu.counter.v1", "type": "counter",
                   "counters": {"drawlists": 1, "tcs_invocations": 0, "ds_invocations": 0},
                   "drawlist_stats": [{"drawlist": 0, "draw_id": 0, "vs": empty, "fs": empty}]}
        record = counter_record_from_message(message)
        self.assertIsNone(record.drawlist_stats[0].tessellation_control)
        self.assertIsNone(record.drawlist_stats[0].tessellation_evaluation)
        message["counters"]["tcs_invocations"] = 1
        with self.assertRaises(CounterProtocolError):
            counter_record_from_message(message)

    def test_geometry_drawlist_is_preserved_and_cross_checked(self) -> None:
        empty = {"invocations": 0, "program": {"groups": 0, "instructions": 0,
                 "alu": 0, "tex": 0, "memory": 0},
                 "executed": {"alu": 0, "tex": 0, "memory": 0}}
        gs = {"invocations": 2, "program": {"groups": 4, "instructions": 4,
              "alu": 1, "tex": 1, "memory": 2},
              "executed": {"alu": 2, "tex": 2, "memory": 4}}
        message = {"schema": "pvrgpu.counter.v1", "type": "counter",
                   "counters": {"drawlists": 1, "gs_invocations": 2,
                                "gs_alu_instructions": 2, "gs_tex_instructions": 2,
                                "gs_memory_instructions": 4},
                   "drawlist_stats": [{"drawlist": 0, "draw_id": 0,
                                       "vs": empty, "fs": empty, "gs": gs}]}
        record = counter_record_from_message(message)
        self.assertEqual(record.drawlist_stats[0].geometry.invocations, 2)
        self.assertEqual(record.drawlist_stats[0].geometry.executed_tex_instructions, 2)
        self.assertIn("gs_tex_instructions", ALL_COUNTER_FIELDS)
        self.assertEqual(COUNTER_INFO["gs_tex_instructions"][1], "instructions")
        for field in ("gs_invocations", "gs_alu_instructions", "gs_tex_instructions", "gs_memory_instructions"):
            message["counters"][field] += 1
            with self.assertRaises(CounterProtocolError):
                counter_record_from_message(message)
            message["counters"][field] -= 1
        del message["drawlist_stats"][0]["gs"]
        with self.assertRaises(CounterProtocolError):
            counter_record_from_message(message)

    def test_cache_bypass_hello_requires_json_boolean(self) -> None:
        accepted = parse_jsonl_line(
            '{"schema":"pvrgpu.counter.v1","type":"hello",'
            '"cache_bypass":false,"memory_mode":"cache",'
            '"cache_simulated":true}'
        )
        self.assertIs(accepted["cache_bypass"], False)
        self.assertEqual(accepted["memory_mode"], "cache")
        self.assertIs(accepted["cache_simulated"], True)
        with self.assertRaises(CounterProtocolError):
            parse_jsonl_line(
                '{"schema":"pvrgpu.counter.v1","type":"hello",'
                '"cache_bypass":"off"}'
            )
        with self.assertRaises(CounterProtocolError):
            parse_jsonl_line(
                '{"schema":"pvrgpu.counter.v1","type":"hello",'
                '"memory_mode":"fast"}'
            )
        with self.assertRaises(CounterProtocolError):
            parse_jsonl_line(
                '{"schema":"pvrgpu.counter.v1","type":"hello",'
                '"cache_simulated":"yes"}'
            )

    def test_memory_path_counter_catalog_is_complete(self) -> None:
        memory_fields = {
            "pixel_data_master_transactions",
            "pixel_data_master_bytes",
            "pixel_data_master_cycles",
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
        self.assertTrue(memory_fields.issubset(MODEL_COUNTER_FIELDS))
        self.assertTrue(memory_fields.issubset(COUNTER_INFO))
        self.assertEqual(len(ALL_COUNTER_FIELDS), len(set(ALL_COUNTER_FIELDS)))

    def test_blend_fixed_function_counter_catalog_is_complete(self) -> None:
        blend_fields = {
            "pbe_color_reads",
            "pbe_blended_fragments",
            "pbe_fragment_writes",
        }
        self.assertTrue(blend_fields.issubset(MODEL_COUNTER_FIELDS))
        self.assertTrue(blend_fields.issubset(COUNTER_INFO))
        self.assertTrue(
            all(COUNTER_INFO[field][1] == "fragments" for field in blend_fields)
        )

    def test_vertex_attribute_counter_catalog_is_complete(self) -> None:
        attribute_fields = {
            "vertex_attribute_fetches",
            "vertex_attribute_bytes",
        }
        self.assertTrue(attribute_fields.issubset(MODEL_COUNTER_FIELDS))
        self.assertTrue(attribute_fields.issubset(COUNTER_INFO))
        self.assertEqual(COUNTER_INFO["vertex_attribute_fetches"][1], "fetches")
        self.assertEqual(COUNTER_INFO["vertex_attribute_bytes"][1], "bytes")

    def test_varying_coefficient_counter_catalog_has_exact_units(self) -> None:
        expected_units = {
            "parameter_coefficient_sets": "sets",
            "parameter_write_bytes": "bytes",
            "pds_coefficient_tasks": "tasks",
            "pds_douti_issues": "issues",
            "usc_coefficient_load_bytes": "bytes",
        }
        self.assertTrue(expected_units.keys() <= set(MODEL_COUNTER_FIELDS))
        self.assertTrue(expected_units.keys() <= set(COUNTER_INFO))
        self.assertEqual(
            {
                field: COUNTER_INFO[field][1]
                for field in expected_units
            },
            expected_units,
        )

    def test_parses_llvmpipe_markdown(self) -> None:
        markdown = """# report
- Renderer: `llvmpipe`

| Frame | Marker | ia_vertices | ps_invocations | drawlists |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 7 | 4 | 65536 | 1 |
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Report.md"
            path.write_text(markdown)
            report = parse_markdown_report(path)
        self.assertEqual(report.metadata["Renderer"], "llvmpipe")
        self.assertEqual(report.records[0].values["ps_invocations"], 65536)
        self.assertEqual(report.records[0].provenance, "reported")

    def test_parses_model_jsonl(self) -> None:
        line = json.dumps(
            {
                "schema": "pvrgpu.counter.v1",
                "type": "counter",
                "source": "fixture",
                "provenance": "mock",
                "frame": 2,
                "counters": {
                    "virtual_gpu_cycles": 1234,
                    "drawlists": 1,
                    "vs_alu_instructions": 0,
                    "vs_tex_instructions": 0,
                    "vs_memory_instructions": 16,
                    "fs_alu_instructions": 64,
                    "fs_tex_instructions": 0,
                    "fs_memory_instructions": 0,
                    "pbe_color_reads": 16,
                    "pbe_blended_fragments": 16,
                    "pbe_fragment_writes": 16,
                },
                "drawlist_stats": [
                    {
                        "drawlist": 0,
                        "draw_id": 7,
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
                            "invocations": 16,
                            "program": {
                                "groups": 5,
                                "instructions": 5,
                                "alu": 4,
                                "tex": 0,
                                "memory": 0,
                            },
                            "executed": {"alu": 64, "tex": 0, "memory": 0},
                        },
                    }
                ],
            }
        )
        message = parse_jsonl_line(line)
        record = counter_record_from_message(message)
        self.assertEqual(record.frame, 2)
        self.assertEqual(record.values["virtual_gpu_cycles"], 1234)
        self.assertEqual(record.values["pbe_color_reads"], 16)
        self.assertEqual(record.values["pbe_blended_fragments"], 16)
        self.assertEqual(record.values["pbe_fragment_writes"], 16)
        self.assertEqual(record.drawlist_stats[0].draw_id, 7)
        self.assertEqual(
            record.drawlist_stats[0].vertex.executed_memory_instructions, 16
        )
        self.assertEqual(
            record.drawlist_stats[0].fragment.executed_alu_instructions, 64
        )

    def test_rejects_mismatched_drawlist_aggregate(self) -> None:
        message = {
            "schema": "pvrgpu.counter.v1",
            "type": "counter",
            "counters": {"drawlists": 1, "vs_alu_instructions": 3},
            "drawlist_stats": [
                {
                    "drawlist": 0,
                    "draw_id": 0,
                    "vs": {
                        "invocations": 1,
                        "program": {
                            "groups": 1,
                            "instructions": 1,
                            "alu": 1,
                            "tex": 0,
                            "memory": 0,
                        },
                        "executed": {"alu": 2, "tex": 0, "memory": 0},
                    },
                    "fs": {
                        "invocations": 0,
                        "program": {
                            "groups": 0,
                            "instructions": 0,
                            "alu": 0,
                            "tex": 0,
                            "memory": 0,
                        },
                        "executed": {"alu": 0, "tex": 0, "memory": 0},
                    },
                }
            ],
        }
        with self.assertRaises(CounterProtocolError):
            counter_record_from_message(message)

    def test_rejects_wrong_schema(self) -> None:
        with self.assertRaises(CounterProtocolError):
            parse_jsonl_line('{"schema":"bad","type":"counter"}')

    def test_rejects_oversize_record(self) -> None:
        with self.assertRaises(CounterProtocolError):
            parse_jsonl_line(b"{" + b" " * (1024 * 1024) + b"}")


if __name__ == "__main__":
    unittest.main()
