from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from counter_protocol import CounterProtocolError, STANDARD_COUNTER_FIELDS  # noqa: E402
from rdc.write_counter_txt import (  # noqa: E402
    counters_from_golden_report,
    counters_from_pvrgpu_jsonl,
    format_counter_text,
)


SCRIPT = TOOLS_DIR / "rdc" / "write_counter_txt.py"


def sample_counters() -> dict[str, int]:
    return {
        field: index
        for index, field in enumerate(STANDARD_COUNTER_FIELDS)
    }


def golden_markdown(
    rows: list[dict[str, int | float]], *, omitted_field: str | None = None
) -> str:
    fields = [
        field for field in STANDARD_COUNTER_FIELDS if field != omitted_field
    ]
    headers = ["Frame", "Marker", *fields]
    lines = [
        "# Golden report",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---:" for _ in headers) + " |",
    ]
    for frame, counters in enumerate(rows, start=1):
        values = [str(frame), str(frame), *(str(counters[field]) for field in fields)]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def pvrgpu_messages(
    counters: dict[str, object] | None = None,
    *,
    formal_mesa_ingest: bool = False,
) -> list[dict[str, object]]:
    hello: dict[str, object] = {
        "schema": "pvrgpu.counter.v1",
        "type": "hello",
        "backend": "pvrgpu",
    }
    if formal_mesa_ingest:
        hello.update(
            {
                "mesa_command_ingest": True,
                "command_source": "renderdoc-mesa-gallium-trace-poc",
                "mesa_command_schema": "pvrgpu.mesa-poc-command.v1",
                "rdc_sha256": "1" * 64,
                "api_trace_sha256": "a" * 64,
                "gallium_trace_sha256": "f" * 64,
            }
        )
    return [
        hello,
        {
            "schema": "pvrgpu.counter.v1",
            "type": "counter",
            "frame": 1,
            "source": "pvrgpu-systemc",
            "provenance": "modeled",
            "counters": counters if counters is not None else sample_counters(),
        },
        {
            "schema": "pvrgpu.counter.v1",
            "type": "done",
            "frames": 1,
            "pool_leaks": 0,
        },
    ]


def write_jsonl(
    path: Path,
    messages: list[dict[str, object]],
    *,
    include_model_noise: bool = False,
) -> None:
    text = "".join(json.dumps(message) + "\n" for message in messages)
    if include_model_noise:
        text = (
            "SystemC 3.0.1 --- simulator banner\n"
            + text
            + "@CAPTURE: fill_solid sample=1 png=frame.png\n"
            + "Info: /OSCI/SystemC: Simulation stopped by user.\n"
        )
    path.write_text(text, encoding="utf-8")


class RdcCounterTextTests(unittest.TestCase):
    def test_golden_report_writes_exact_fixed_order(self) -> None:
        expected = sample_counters()
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "Report.md"
            report.write_text(golden_markdown([expected]), encoding="utf-8")
            parsed = counters_from_golden_report(report)

        self.assertEqual(parsed, expected)
        output = format_counter_text(parsed)
        self.assertEqual(
            output.splitlines(),
            [f"{field}={expected[field]}" for field in STANDARD_COUNTER_FIELDS],
        )
        self.assertTrue(output.endswith("\n"))

    def test_output_option_writes_file_instead_of_stdout(self) -> None:
        expected = sample_counters()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "Report.md"
            output = root / "nested" / "counter.txt"
            report.write_text(golden_markdown([expected]), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--golden-report",
                    str(report),
                    "--output",
                    str(output),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, "")
            self.assertEqual(output.read_text(encoding="utf-8"), format_counter_text(expected))

    def test_golden_report_requires_one_complete_integer_frame(self) -> None:
        base = sample_counters()
        invalid_reports = {
            "two frames": golden_markdown([base, base]),
            "missing field": golden_markdown([base], omitted_field="texel_fetches"),
            "negative": golden_markdown(
                [{**base, "ps_invocations": -1}]
            ),
            "floating point": golden_markdown(
                [{**base, "ps_invocations": 1.0}]
            ),
        }
        for description, text in invalid_reports.items():
            with self.subTest(description=description):
                with tempfile.TemporaryDirectory() as directory:
                    report = Path(directory) / "Report.md"
                    report.write_text(text, encoding="utf-8")
                    with self.assertRaises(CounterProtocolError):
                        counters_from_golden_report(report)

    def test_pvrgpu_jsonl_requires_modeled_counter_and_clean_done(self) -> None:
        expected = sample_counters()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stdout.jsonl"
            write_jsonl(path, pvrgpu_messages(), include_model_noise=True)
            parsed = counters_from_pvrgpu_jsonl(path)
        self.assertEqual(parsed, expected)

    def test_pvrgpu_jsonl_formal_mesa_ingest_accepts_complete_evidence(self) -> None:
        expected = sample_counters()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stdout.jsonl"
            write_jsonl(path, pvrgpu_messages(formal_mesa_ingest=True))
            parsed = counters_from_pvrgpu_jsonl(
                path, require_mesa_ingest=True
            )
        self.assertEqual(parsed, expected)

    def test_pvrgpu_jsonl_formal_mesa_ingest_rejects_bad_evidence(self) -> None:
        mutations = {
            "ingest disabled": ("mesa_command_ingest", False),
            "wrong command source": ("command_source", "builtin-glbench-fixture"),
            "wrong command schema": ("mesa_command_schema", "pvrgpu.command.v0"),
            "invalid RDC digest": ("rdc_sha256", "not-a-sha256"),
            "invalid API trace digest": ("api_trace_sha256", "A" * 64),
            "invalid Gallium trace digest": ("gallium_trace_sha256", "f" * 63),
        }
        for description, (field, value) in mutations.items():
            with self.subTest(description=description):
                messages = pvrgpu_messages(formal_mesa_ingest=True)
                messages[0][field] = value
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "stdout.jsonl"
                    write_jsonl(path, messages)
                    with self.assertRaises(CounterProtocolError):
                        counters_from_pvrgpu_jsonl(
                            path, require_mesa_ingest=True
                        )

        messages = pvrgpu_messages(formal_mesa_ingest=True)
        del messages[0]["rdc_sha256"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stdout.jsonl"
            write_jsonl(path, messages)
            with self.assertRaises(CounterProtocolError):
                counters_from_pvrgpu_jsonl(path, require_mesa_ingest=True)

    def test_pvrgpu_jsonl_rejects_invalid_acceptance_evidence(self) -> None:
        base = pvrgpu_messages()

        duplicate_counter = copy.deepcopy(base)
        duplicate_counter.insert(2, copy.deepcopy(base[1]))

        wrong_provenance = copy.deepcopy(base)
        wrong_provenance[1]["provenance"] = "reported"

        missing_hello = copy.deepcopy(base[1:])

        wrong_hello_backend = copy.deepcopy(base)
        wrong_hello_backend[0]["backend"] = "llvmpipe"

        leaking_done = copy.deepcopy(base)
        leaking_done[2]["pool_leaks"] = 1

        missing_done = copy.deepcopy(base[:-1])

        wrong_frame_count = copy.deepcopy(base)
        wrong_frame_count[2]["frames"] = 2

        negative_counter = copy.deepcopy(base)
        negative_counter[1]["counters"]["ps_invocations"] = -1  # type: ignore[index]

        floating_counter = copy.deepcopy(base)
        floating_counter[1]["counters"]["ps_invocations"] = 1.0  # type: ignore[index]

        boolean_counter = copy.deepcopy(base)
        boolean_counter[1]["counters"]["ps_invocations"] = True  # type: ignore[index]

        missing_counter_field = copy.deepcopy(base)
        del missing_counter_field[1]["counters"]["texel_fetches"]  # type: ignore[index]

        protocol_error = copy.deepcopy(base)
        protocol_error.insert(
            1,
            {
                "schema": "pvrgpu.counter.v1",
                "type": "error",
                "error": "shader decode failed",
            },
        )

        invalid_streams = {
            "duplicate counter": duplicate_counter,
            "wrong provenance": wrong_provenance,
            "missing hello": missing_hello,
            "wrong hello backend": wrong_hello_backend,
            "pool leak": leaking_done,
            "missing done": missing_done,
            "wrong frame count": wrong_frame_count,
            "negative counter": negative_counter,
            "floating counter": floating_counter,
            "boolean counter": boolean_counter,
            "missing counter field": missing_counter_field,
            "protocol error": protocol_error,
        }
        for description, messages in invalid_streams.items():
            with self.subTest(description=description):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "stdout.jsonl"
                    write_jsonl(path, messages)
                    with self.assertRaises(CounterProtocolError):
                        counters_from_pvrgpu_jsonl(path)

    def test_cli_requires_exactly_one_input_kind(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "Report.md"
            jsonl = root / "stdout.jsonl"
            report.write_text(
                golden_markdown([sample_counters()]), encoding="utf-8"
            )
            write_jsonl(jsonl, pvrgpu_messages())
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--golden-report",
                    str(report),
                    "--pvrgpu-jsonl",
                    str(jsonl),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("not allowed with argument", completed.stderr)

    def test_cli_require_mesa_ingest_accepts_only_formal_pvrgpu_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl = root / "stdout.jsonl"
            write_jsonl(jsonl, pvrgpu_messages(formal_mesa_ingest=True))
            accepted = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--pvrgpu-jsonl",
                    str(jsonl),
                    "--require-mesa-ingest",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )

            report = root / "Report.md"
            report.write_text(
                golden_markdown([sample_counters()]), encoding="utf-8"
            )
            rejected = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--golden-report",
                    str(report),
                    "--require-mesa-ingest",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )

        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertEqual(accepted.stdout, format_counter_text(sample_counters()))
        self.assertEqual(rejected.returncode, 2)
        self.assertIn(
            "--require-mesa-ingest requires --pvrgpu-jsonl", rejected.stderr
        )


if __name__ == "__main__":
    unittest.main()
