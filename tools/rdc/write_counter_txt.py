#!/usr/bin/env python3
"""Write one frame's normalized 17-field counters as name=value text."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from counter_protocol import (  # noqa: E402
    CounterProtocolError,
    STANDARD_COUNTER_FIELDS,
    counter_record_from_message,
    parse_jsonl_line,
    parse_markdown_report,
)


MESA_POC_COMMAND_SOURCE = "renderdoc-mesa-gallium-trace-poc"
MESA_POC_COMMAND_SCHEMA = "pvrgpu.mesa-poc-command.v1"
MESA_POC_SHA256_FIELDS = (
    "rdc_sha256",
    "api_trace_sha256",
    "gallium_trace_sha256",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _standard_integer_counters(
    values: Mapping[str, int | float], *, source: str
) -> dict[str, int]:
    missing = [field for field in STANDARD_COUNTER_FIELDS if field not in values]
    if missing:
        raise CounterProtocolError(
            f"{source} is missing standard counter fields: {', '.join(missing)}"
        )

    counters: dict[str, int] = {}
    for field in STANDARD_COUNTER_FIELDS:
        value = values[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CounterProtocolError(
                f"{source}.{field} must be a non-negative integer"
            )
        counters[field] = value
    return counters


def counters_from_golden_report(report_path: str | Path) -> dict[str, int]:
    report = parse_markdown_report(report_path)
    if len(report.records) != 1:
        raise CounterProtocolError(
            "Golden report must contain exactly one frame, "
            f"found {len(report.records)}"
        )
    return _standard_integer_counters(
        report.records[0].values, source="Golden report"
    )


def _validate_mesa_ingest_hello(hello: Mapping[str, object]) -> None:
    if hello.get("mesa_command_ingest") is not True:
        raise CounterProtocolError(
            "PvrGPU hello.mesa_command_ingest must be true when Mesa ingest "
            "evidence is required"
        )
    if hello.get("command_source") != MESA_POC_COMMAND_SOURCE:
        raise CounterProtocolError(
            "PvrGPU hello.command_source must be "
            f"'{MESA_POC_COMMAND_SOURCE}' when Mesa ingest evidence is required"
        )
    if hello.get("mesa_command_schema") != MESA_POC_COMMAND_SCHEMA:
        raise CounterProtocolError(
            "PvrGPU hello.mesa_command_schema must be "
            f"'{MESA_POC_COMMAND_SCHEMA}' when Mesa ingest evidence is required"
        )
    for field in MESA_POC_SHA256_FIELDS:
        value = hello.get(field)
        if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
            raise CounterProtocolError(
                f"PvrGPU hello.{field} must be a lowercase SHA-256 hex digest "
                "when Mesa ingest evidence is required"
            )


def counters_from_pvrgpu_jsonl(
    jsonl_path: str | Path,
    *,
    require_mesa_ingest: bool = False,
    expected_rdc_sha256: str | None = None,
) -> dict[str, int]:
    path = Path(jsonl_path)
    messages: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        protocol_line = line.lstrip()
        # A captured model stdout also contains the SystemC banner/info and
        # optional @CAPTURE lines. The protocol boundary used by the existing
        # integration checker is an object-shaped line after left trimming.
        if not protocol_line.startswith("{"):
            continue
        try:
            messages.append(parse_jsonl_line(protocol_line))
        except CounterProtocolError as exc:
            raise CounterProtocolError(
                f"PvrGPU JSONL line {line_number}: {exc}"
            ) from exc

    errors = [message for message in messages if message.get("type") == "error"]
    if errors:
        raise CounterProtocolError(
            "PvrGPU reported an error: "
            f"{errors[0].get('error', 'unspecified protocol error')}"
        )

    allowed_types = {"hello", "counter", "done"}
    unexpected = [
        message.get("type")
        for message in messages
        if message.get("type") not in allowed_types
    ]
    if unexpected:
        raise CounterProtocolError(
            f"PvrGPU JSONL contains unsupported message type: {unexpected[0]!r}"
        )

    hello_messages = [
        message for message in messages if message.get("type") == "hello"
    ]
    if len(hello_messages) != 1:
        raise CounterProtocolError(
            "PvrGPU JSONL must contain exactly one hello message, "
            f"found {len(hello_messages)}"
        )
    hello = hello_messages[0]
    if hello.get("backend") != "pvrgpu":
        raise CounterProtocolError("PvrGPU hello.backend must be 'pvrgpu'")
    if require_mesa_ingest:
        _validate_mesa_ingest_hello(hello)
    if expected_rdc_sha256 is not None:
        if SHA256_PATTERN.fullmatch(expected_rdc_sha256) is None:
            raise CounterProtocolError(
                "Expected RDC SHA-256 must be a lowercase 64-digit hex digest"
            )
        if hello.get("rdc_sha256") != expected_rdc_sha256:
            raise CounterProtocolError(
                "PvrGPU hello.rdc_sha256 does not match the RDC under test"
            )

    counter_messages = [
        message for message in messages if message.get("type") == "counter"
    ]
    if len(counter_messages) != 1:
        raise CounterProtocolError(
            "PvrGPU JSONL must contain exactly one counter, "
            f"found {len(counter_messages)}"
        )

    done_messages = [
        message for message in messages if message.get("type") == "done"
    ]
    if len(done_messages) != 1:
        raise CounterProtocolError(
            "PvrGPU JSONL must contain exactly one done message, "
            f"found {len(done_messages)}"
        )

    counter_message = counter_messages[0]
    frame = counter_message.get("frame")
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise CounterProtocolError(
            "PvrGPU counter.frame must be a non-negative integer"
        )
    if counter_message.get("provenance") != "modeled":
        raise CounterProtocolError("PvrGPU counter provenance must be 'modeled'")

    done = done_messages[0]
    frames = done.get("frames")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames != 1:
        raise CounterProtocolError("PvrGPU done.frames must be exactly 1")
    pool_leaks = done.get("pool_leaks")
    if isinstance(pool_leaks, bool) or not isinstance(pool_leaks, int):
        raise CounterProtocolError(
            "PvrGPU done.pool_leaks must be the integer 0"
        )
    if pool_leaks != 0:
        raise CounterProtocolError(
            f"PvrGPU done.pool_leaks must be 0, got {pool_leaks}"
        )

    record = counter_record_from_message(counter_message)
    return _standard_integer_counters(record.values, source="PvrGPU counter")


def format_counter_text(counters: Mapping[str, int]) -> str:
    validated = _standard_integer_counters(counters, source="Counter output")
    return "".join(
        f"{field}={validated[field]}\n" for field in STANDARD_COUNTER_FIELDS
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write one strict 17-field RDC counter text file."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--golden-report",
        type=Path,
        metavar="REPORT.md",
        help="Mesa llvmpipe Golden Markdown report",
    )
    source.add_argument(
        "--pvrgpu-jsonl",
        type=Path,
        metavar="STDOUT.jsonl",
        help="PvrGPU JSONL stdout containing one counter and one done message",
    )
    parser.add_argument(
        "--output",
        type=Path,
        metavar="COUNTER.txt",
        help="write to this file instead of stdout",
    )
    parser.add_argument(
        "--require-mesa-ingest",
        action="store_true",
        help=(
            "require formal RenderDoc + Mesa/Gallium command-ingest provenance "
            "in the PvrGPU hello message"
        ),
    )
    parser.add_argument(
        "--expected-rdc-sha256",
        help="require PvrGPU hello.rdc_sha256 to match this exact digest",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    options = parser.parse_args(argv)
    if options.require_mesa_ingest and options.pvrgpu_jsonl is None:
        parser.error("--require-mesa-ingest requires --pvrgpu-jsonl")
    try:
        if options.golden_report is not None:
            counters = counters_from_golden_report(options.golden_report)
        else:
            counters = counters_from_pvrgpu_jsonl(
                options.pvrgpu_jsonl,
                require_mesa_ingest=options.require_mesa_ingest,
                expected_rdc_sha256=options.expected_rdc_sha256,
            )
        output = format_counter_text(counters)
        if options.output is None:
            sys.stdout.write(output)
        else:
            options.output.parent.mkdir(parents=True, exist_ok=True)
            options.output.write_text(output, encoding="utf-8")
    except (CounterProtocolError, OSError, UnicodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
