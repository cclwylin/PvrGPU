from __future__ import annotations

import json
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
    def test_cache_bypass_hello_requires_json_boolean(self) -> None:
        accepted = parse_jsonl_line(
            '{"schema":"pvrgpu.counter.v1","type":"hello",'
            '"cache_bypass":false}'
        )
        self.assertIs(accepted["cache_bypass"], False)
        with self.assertRaises(CounterProtocolError):
            parse_jsonl_line(
                '{"schema":"pvrgpu.counter.v1","type":"hello",'
                '"cache_bypass":"off"}'
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
