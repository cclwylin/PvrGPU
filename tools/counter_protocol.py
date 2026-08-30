"""Counter protocol shared by the Qt controller and its tests.

The llvmpipe producer is a patched Mesa Markdown report.  The current PvrGPU
functional slice supports four Fill.Solid state cases, three indexed
Triangle.Setup cases, Attribute fetch cases 1/2/4/8, varying cases 1/2/4/8,
and Fill.Texture.Nearest; it emits a versioned JSONL stream and framebuffer PNG from
built-in GLBench fixtures rather than Mesa command ingest. Both counter paths
normalize to CounterRecord so the UI does not need backend-specific widgets.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


PROTOCOL = "pvrgpu-jsonl"
PROTOCOL_VERSION = 1
SCHEMA = "pvrgpu.counter.v1"
MAX_JSONL_BYTES = 1024 * 1024

# Run configuration transported by a producer hello message.  Values remain
# top-level JSON metadata rather than counters so they are never summed across
# frames.  The current boolean field is type-checked by parse_jsonl_line().
HELLO_METADATA_FIELDS: tuple[str, ...] = ("cache_bypass",)

STANDARD_COUNTER_FIELDS: tuple[str, ...] = (
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

MODEL_COUNTER_FIELDS: tuple[str, ...] = (
    "virtual_gpu_cycles",
    "tiler_cycles",
    "renderer_cycles",
    "usc_groups",
    "texture_requests",
    "fifo_stall_events",
    "pool_bytes_in_flight",
    "pool_high_water_bytes",
    "vdm_cycles",
    "vertex_fetch_cycles",
    "vertex_attribute_fetches",
    "vertex_attribute_bytes",
    "pco_decode_cycles",
    "pco_instructions",
    "vs_alu_instructions",
    "vs_tex_instructions",
    "vs_memory_instructions",
    "fs_alu_instructions",
    "fs_tex_instructions",
    "fs_memory_instructions",
    "usc_slot_cycles",
    "usc_cluster_cycles",
    "clip_cull_cycles",
    "tiler_bin_cycles",
    "parameter_buffer_cycles",
    "parameter_coefficient_sets",
    "parameter_write_bytes",
    "pds_coefficient_tasks",
    "pds_douti_issues",
    "usc_coefficient_load_bytes",
    "tile_scheduler_cycles",
    "isp_cycles",
    "fragment_frontend_cycles",
    "texture_cycles",
    "pbe_cycles",
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
    "tiles_binned",
    "tiles_scheduled",
    "covered_pixels",
    "fragment_candidates",
    "hsr_rejected_fragments",
    "depth_tested_fragments",
    "depth_rejected_fragments",
    "depth_written_fragments",
    "pbe_color_reads",
    "pbe_blended_fragments",
    "pbe_fragment_writes",
    "pbe_pixels_written",
)

ALL_COUNTER_FIELDS = STANDARD_COUNTER_FIELDS + MODEL_COUNTER_FIELDS

COUNTER_INFO: Mapping[str, tuple[str, str, str]] = {
    "ia_vertices": ("Input vertices", "count", "Input-assembler vertices."),
    "ia_primitives": ("Input primitives", "count", "Input-assembler primitives."),
    "vs_invocations": ("Vertex shader", "invocations", "Vertex shader invocations."),
    "gs_invocations": ("Geometry shader", "invocations", "Geometry shader invocations."),
    "gs_primitives": ("GS primitives", "count", "Geometry shader output primitives."),
    "c_invocations": ("Clip invocations", "count", "Primitives sent to clip/raster."),
    "c_primitives": ("Clip primitives", "count", "Primitives entering setup."),
    "ps_invocations": ("Fragment shader", "invocations", "Active fragment shader lanes."),
    "hs_invocations": ("Hull shader", "invocations", "Tessellation-control invocations."),
    "ds_invocations": ("Domain shader", "invocations", "Tessellation-evaluation invocations."),
    "cs_invocations": ("Compute shader", "invocations", "Compute shader invocations."),
    "ts_invocations": ("Task shader", "invocations", "Task shader invocations."),
    "ms_invocations": ("Mesh shader", "invocations", "Mesh shader invocations."),
    "ms_primitives": ("Mesh primitives", "count", "Mesh shader output primitives."),
    "drawlists": (
        "DrawLists",
        "count",
        "DrawList submissions in the current frame.",
    ),
    "setup_triangles": ("Setup triangles", "count", "Triangles entering setup before culling."),
    "texel_fetches": ("Texel fetches", "lanes", "Sampler fetch lanes, including filtering."),
    "virtual_gpu_cycles": ("Virtual GPU cycles", "cycles", "Modeled cycle-equivalent critical path."),
    "tiler_cycles": ("Tiler cycles", "cycles", "Modeled geometry/tiler service cycles."),
    "renderer_cycles": ("Renderer cycles", "cycles", "Modeled fragment/renderer service cycles."),
    "usc_groups": ("USC groups", "groups", "Modeled shader-lane issue groups."),
    "texture_requests": ("Texture requests", "requests", "Modeled logical texture requests."),
    "fifo_stall_events": ("FIFO stalls", "events", "Modeled bounded-FIFO backpressure events."),
    "pool_bytes_in_flight": ("Pool bytes in flight", "bytes", "Live MemoryPool payload bytes."),
    "pool_high_water_bytes": ("Pool high-water", "bytes", "Peak MemoryPool payload bytes."),
    "vdm_cycles": ("VDM", "cycles", "Modeled VDM service cycles."),
    "vertex_fetch_cycles": ("Vertex fetch", "cycles", "Modeled vertex-fetch service cycles."),
    "vertex_attribute_fetches": (
        "Vertex attributes",
        "fetches",
        "Vertex-attribute records fetched on post-transform cache misses.",
    ),
    "vertex_attribute_bytes": (
        "Vertex attribute bytes",
        "bytes",
        "Source VBO bytes fetched; GLES default components are excluded.",
    ),
    "pco_decode_cycles": ("PCO decode", "cycles", "Modeled vertex and fragment PCO decode cycles."),
    "pco_instructions": (
        "PCO instructions",
        "instructions",
        "Semantic public-PCO instructions decoded from vertex and fragment binaries.",
    ),
    "vs_alu_instructions": (
        "VS ALU instructions",
        "instructions",
        "Repeat-expanded vertex ALU/move instructions executed across all lanes.",
    ),
    "vs_tex_instructions": (
        "VS Tex instructions",
        "instructions",
        "Repeat-expanded vertex texture instructions executed across all lanes.",
    ),
    "vs_memory_instructions": (
        "VS Memory instructions",
        "instructions",
        "Repeat-expanded vertex memory/export instructions executed across all lanes.",
    ),
    "fs_alu_instructions": (
        "FS ALU instructions",
        "instructions",
        "Repeat-expanded fragment ALU/move instructions executed across all lanes.",
    ),
    "fs_tex_instructions": (
        "FS Tex instructions",
        "instructions",
        "Repeat-expanded fragment texture instructions executed across all lanes.",
    ),
    "fs_memory_instructions": (
        "FS Memory instructions",
        "instructions",
        "Repeat-expanded fragment memory/export instructions executed across all lanes.",
    ),
    "usc_slot_cycles": ("USC slot", "cycles", "Modeled USC slot issue cycles."),
    "usc_cluster_cycles": ("USC cluster", "cycles", "Modeled USC cluster execution cycles."),
    "clip_cull_cycles": ("Clip/cull", "cycles", "Modeled clip and cull cycles."),
    "tiler_bin_cycles": ("Tiler binning", "cycles", "Modeled tile-binning cycles."),
    "parameter_buffer_cycles": ("Parameter buffer", "cycles", "Modeled parameter-buffer service cycles."),
    "parameter_coefficient_sets": (
        "Parameter coefficient sets",
        "sets",
        "Four-dword A/B/C/PAD interpolation coefficient sets written per primitive.",
    ),
    "parameter_write_bytes": (
        "Parameter writes",
        "bytes",
        "Interpolation coefficient payload bytes written to the on-chip parameter buffer.",
    ),
    "pds_coefficient_tasks": (
        "PDS coefficient tasks",
        "tasks",
        "Fragment-quad coefficient DMA tasks dispatched by PDS.",
    ),
    "pds_douti_issues": (
        "PDS DOUTI issues",
        "issues",
        "PDS DOUTI coefficient-transfer issues sent to USC.",
    ),
    "usc_coefficient_load_bytes": (
        "USC coefficient loads",
        "bytes",
        "Coefficient-bank bytes loaded into USC fragment tasks.",
    ),
    "tile_scheduler_cycles": ("Tile scheduler", "cycles", "Modeled fragment tile-scheduling cycles."),
    "isp_cycles": ("ISP", "cycles", "Modeled ISP/HSR service cycles."),
    "fragment_frontend_cycles": ("Fragment frontend", "cycles", "Modeled fragment-frontend service cycles."),
    "texture_cycles": ("TPU", "cycles", "Modeled texture-processing service cycles."),
    "pbe_cycles": ("PBE", "cycles", "Modeled pixel-backend service cycles."),
    "pixel_data_master_transactions": (
        "PixelDM transactions",
        "transactions",
        "Framebuffer transactions issued by the Pixel Data Master toward SLC.",
    ),
    "pixel_data_master_bytes": (
        "PixelDM bytes",
        "bytes",
        "Framebuffer payload bytes issued by the Pixel Data Master toward SLC.",
    ),
    "pixel_data_master_cycles": (
        "PixelDM",
        "cycles",
        "Modeled event-driven Pixel Data Master service cycles.",
    ),
    "tcu_line_accesses": (
        "TCU line accesses",
        "lines",
        "Texture Cache Unit cache-line lookups.",
    ),
    "tcu_read_accesses": (
        "TCU reads",
        "lines",
        "Texture Cache Unit cache-line read accesses.",
    ),
    "tcu_hits": (
        "TCU hits",
        "lines",
        "Texture cache lookups satisfied by resident TCU lines.",
    ),
    "tcu_misses": (
        "TCU misses",
        "lines",
        "Texture cache lookups requiring SLC service.",
    ),
    "tcu_evictions": (
        "TCU evictions",
        "lines",
        "Valid TCU cache lines displaced by true-LRU replacement.",
    ),
    "tcu_writebacks": (
        "TCU writebacks",
        "lines",
        "Dirty TCU lines written to the lower cache; texture reads leave this zero.",
    ),
    "tcu_bypassed": (
        "TCU bypassed",
        "lines",
        "Texture lines forwarded without TCU lookup or allocation.",
    ),
    "tcu_cycles": (
        "TCU",
        "cycles",
        "Event-driven Texture Cache Unit lookup service cycles.",
    ),
    "slc_line_accesses": (
        "SLC line accesses",
        "lines",
        "Cache-line lookups actually performed by the System Level Cache.",
    ),
    "slc_read_accesses": (
        "SLC reads",
        "lines",
        "SLC cache-line read accesses.",
    ),
    "slc_write_accesses": (
        "SLC writes",
        "lines",
        "SLC cache-line write accesses.",
    ),
    "slc_hits": (
        "SLC hits",
        "lines",
        "SLC cache-line lookups satisfied by a resident line.",
    ),
    "slc_misses": (
        "SLC misses",
        "lines",
        "SLC cache-line lookups that required lower-memory service.",
    ),
    "slc_evictions": (
        "SLC evictions",
        "lines",
        "Valid SLC cache lines displaced by replacement.",
    ),
    "slc_writebacks": (
        "SLC writebacks",
        "lines",
        "Dirty SLC cache lines written back to DRAM.",
    ),
    "slc_bypassed": (
        "SLC bypassed",
        "transactions",
        "Transactions forwarded to DRAM without SLC lookup or allocation.",
    ),
    "slc_cycles": (
        "SLC",
        "cycles",
        "Modeled event-driven SLC service cycles.",
    ),
    "dram_read_transactions": (
        "DRAM reads",
        "transactions",
        "Read transactions serviced by the DRAM model.",
    ),
    "dram_write_transactions": (
        "DRAM writes",
        "transactions",
        "Write transactions serviced by the DRAM model.",
    ),
    "dram_read_bytes": (
        "DRAM read bytes",
        "bytes",
        "Payload bytes returned by the DRAM model.",
    ),
    "dram_write_bytes": (
        "DRAM write bytes",
        "bytes",
        "Payload bytes committed by the DRAM model.",
    ),
    "dram_cycles": (
        "DRAM",
        "cycles",
        "Modeled fixed-latency DRAM service cycles.",
    ),
    "framebuffer_dram_readback_bytes": (
        "Framebuffer DRAM readback",
        "bytes",
        "Final framebuffer bytes read back from the DRAM model for PNG publication.",
    ),
    "tiles_binned": (
        "Tiles binned",
        "tiles",
        "Row-major 32x32 tile records produced by the functional tiler.",
    ),
    "tiles_scheduled": (
        "Tiles scheduled",
        "tiles",
        "Non-empty 32x32 tile records submitted to ISP.",
    ),
    "covered_pixels": (
        "Covered pixels",
        "pixels",
        "Unique framebuffer pixels covered by ISP.",
    ),
    "fragment_candidates": (
        "Fragment candidates",
        "samples",
        "Primitive/sample candidates evaluated by ISP before HSR owner resolution.",
    ),
    "hsr_rejected_fragments": (
        "HSR rejected",
        "samples",
        "Covered candidates eliminated before fragment USC execution.",
    ),
    "depth_tested_fragments": (
        "Depth tested",
        "samples",
        "Covered samples evaluated by the ISP depth comparator.",
    ),
    "depth_rejected_fragments": (
        "Depth rejected",
        "samples",
        "Covered samples rejected by the ISP depth comparator.",
    ),
    "depth_written_fragments": (
        "Depth written",
        "samples",
        "Passing samples that updated the tile-local depth attachment.",
    ),
    "pbe_color_reads": (
        "PBE color reads",
        "fragments",
        "Destination render-target colors read by PBE for fixed-function blending.",
    ),
    "pbe_blended_fragments": (
        "PBE blended fragments",
        "fragments",
        "Fragment outputs evaluated by the enabled fixed-function blend equation.",
    ),
    "pbe_fragment_writes": (
        "PBE fragment writes",
        "fragments",
        "Post-blend fragment colors written to the render target by PBE.",
    ),
    "pbe_pixels_written": (
        "PBE pixels written",
        "pixels",
        "RGBA8 pixels committed to the MemoryPool framebuffer by PBE.",
    ),
}


class CounterProtocolError(ValueError):
    """A counter stream/report is malformed or incompatible."""


@dataclass(frozen=True)
class ShaderInstructionStats:
    invocations: int
    program_groups: int
    program_instructions: int
    program_alu_instructions: int
    program_tex_instructions: int
    program_memory_instructions: int
    executed_alu_instructions: int
    executed_tex_instructions: int
    executed_memory_instructions: int


@dataclass(frozen=True)
class DrawListRecord:
    drawlist: int
    draw_id: int
    vertex: ShaderInstructionStats
    fragment: ShaderInstructionStats


@dataclass(frozen=True)
class CounterRecord:
    frame: int
    marker: str
    values: Mapping[str, int | float]
    source: str
    provenance: str
    virtual_time_ns: int | None = None
    drawlist_stats: tuple[DrawListRecord, ...] = ()


@dataclass(frozen=True)
class CounterReport:
    metadata: Mapping[str, str]
    records: tuple[CounterRecord, ...]

    @property
    def totals(self) -> dict[str, int | float]:
        fields = {key for record in self.records for key in record.values}
        return {
            field: sum(record.values.get(field, 0) for record in self.records)
            for field in fields
        }


def _cells(line: str) -> list[str]:
    return [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]


def _is_alignment(cells: Iterable[str]) -> bool:
    return all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _number(value: Any) -> int | float:
    if isinstance(value, bool):
        raise CounterProtocolError("Boolean is not a counter value")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise CounterProtocolError("NaN/Inf is not allowed")
        return value
    text = str(value).strip().strip("`").replace(",", "")
    if re.fullmatch(r"[-+]?[0-9]+", text):
        return int(text)
    if re.fullmatch(r"[-+]?(?:[0-9]+\.[0-9]*|[0-9]*\.[0-9]+)", text):
        return float(text)
    raise CounterProtocolError(f"Invalid counter value: {value!r}")


def parse_markdown_report(path: str | Path) -> CounterReport:
    report_path = Path(path)
    text = report_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    metadata: dict[str, str] = {}
    for line in lines:
        match = re.match(r"^-\s+([^:]+):\s*(.+?)\s*$", line)
        if match:
            metadata[match.group(1).strip()] = match.group(2).strip().strip("`")

    for header_index, line in enumerate(lines):
        if not line.lstrip().startswith("|"):
            continue
        headers = _cells(line)
        present = [field for field in STANDARD_COUNTER_FIELDS if field in headers]
        if len(present) < 3:
            continue
        if header_index + 1 >= len(lines) or not _is_alignment(_cells(lines[header_index + 1])):
            continue

        records: list[CounterRecord] = []
        for row_line in lines[header_index + 2 :]:
            if not row_line.lstrip().startswith("|"):
                break
            row_cells = _cells(row_line)
            if len(row_cells) != len(headers):
                # A writer may currently be appending this line. Ignore only
                # the incomplete tail so the live UI stays usable.
                break
            row = dict(zip(headers, row_cells, strict=True))
            values = {field: _number(row.get(field, "0")) for field in present}
            try:
                frame = int(str(row.get("Frame", len(records) + 1)).strip())
            except ValueError as exc:
                raise CounterProtocolError("Frame must be an integer") from exc
            records.append(
                CounterRecord(
                    frame=frame,
                    marker=str(row.get("Marker", "")),
                    values=values,
                    source="mesa.llvmpipe.telemetry-patch",
                    provenance="reported",
                )
            )
        if records:
            return CounterReport(metadata=metadata, records=tuple(records))

    raise CounterProtocolError(f"No supported counter table in {report_path}")


def parse_jsonl_line(line: str | bytes) -> dict[str, Any]:
    raw = line.encode("utf-8") if isinstance(line, str) else line
    if len(raw) > MAX_JSONL_BYTES:
        raise CounterProtocolError("JSONL record exceeds 1 MiB")
    try:
        message = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CounterProtocolError(f"Malformed JSONL: {exc}") from exc
    if not isinstance(message, dict):
        raise CounterProtocolError("JSONL record must be an object")

    schema = message.get("schema")
    protocol = message.get("protocol")
    version = message.get("version")
    if schema != SCHEMA and not (protocol == PROTOCOL and version == PROTOCOL_VERSION):
        raise CounterProtocolError("Unsupported counter schema/protocol")
    if not isinstance(message.get("type"), str):
        raise CounterProtocolError("Counter message has no type")
    if (
        message.get("type") == "hello"
        and "cache_bypass" in message
        and not isinstance(message["cache_bypass"], bool)
    ):
        raise CounterProtocolError("hello.cache_bypass must be a JSON boolean")
    return message


def _nonnegative_integer(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CounterProtocolError(f"{description} must be a non-negative integer")
    return value


def _shader_instruction_stats(raw: Any, description: str) -> ShaderInstructionStats:
    if not isinstance(raw, dict):
        raise CounterProtocolError(f"{description} must be an object")
    program = raw.get("program")
    executed = raw.get("executed")
    if not isinstance(program, dict) or not isinstance(executed, dict):
        raise CounterProtocolError(f"{description} has no program/executed object")

    stats = ShaderInstructionStats(
        invocations=_nonnegative_integer(
            raw.get("invocations"), f"{description}.invocations"
        ),
        program_groups=_nonnegative_integer(
            program.get("groups"), f"{description}.program.groups"
        ),
        program_instructions=_nonnegative_integer(
            program.get("instructions"), f"{description}.program.instructions"
        ),
        program_alu_instructions=_nonnegative_integer(
            program.get("alu"), f"{description}.program.alu"
        ),
        program_tex_instructions=_nonnegative_integer(
            program.get("tex"), f"{description}.program.tex"
        ),
        program_memory_instructions=_nonnegative_integer(
            program.get("memory"), f"{description}.program.memory"
        ),
        executed_alu_instructions=_nonnegative_integer(
            executed.get("alu"), f"{description}.executed.alu"
        ),
        executed_tex_instructions=_nonnegative_integer(
            executed.get("tex"), f"{description}.executed.tex"
        ),
        executed_memory_instructions=_nonnegative_integer(
            executed.get("memory"), f"{description}.executed.memory"
        ),
    )
    if (
        stats.program_alu_instructions
        + stats.program_tex_instructions
        + stats.program_memory_instructions
        > stats.program_instructions
    ):
        raise CounterProtocolError(
            f"{description} static ALU/Tex/Memory classes exceed program.instructions"
        )
    # Synchronization/control groups such as public PCO WDF are intentionally
    # part of program.instructions without being mislabeled ALU, texture, or
    # memory. Backend-specific opcode evidence accounts for that remainder.
    return stats


def _drawlist_records(
    message: Mapping[str, Any], values: Mapping[str, int | float]
) -> tuple[DrawListRecord, ...]:
    raw_drawlists = message.get("drawlist_stats")
    if raw_drawlists is None:
        return ()
    if not isinstance(raw_drawlists, list):
        raise CounterProtocolError("drawlist_stats must be an array")

    records: list[DrawListRecord] = []
    for position, raw in enumerate(raw_drawlists):
        if not isinstance(raw, dict):
            raise CounterProtocolError(
                f"drawlist_stats[{position}] must be an object"
            )
        drawlist = _nonnegative_integer(
            raw.get("drawlist"), f"drawlist_stats[{position}].drawlist"
        )
        if drawlist != position:
            raise CounterProtocolError(
                "drawlist_stats indices must be ordered and contiguous"
            )
        records.append(
            DrawListRecord(
                drawlist=drawlist,
                draw_id=_nonnegative_integer(
                    raw.get("draw_id"), f"drawlist_stats[{position}].draw_id"
                ),
                vertex=_shader_instruction_stats(
                    raw.get("vs"), f"drawlist_stats[{position}].vs"
                ),
                fragment=_shader_instruction_stats(
                    raw.get("fs"), f"drawlist_stats[{position}].fs"
                ),
            )
        )

    counter_drawlists = values.get("drawlists")
    if counter_drawlists is not None and counter_drawlists != len(records):
        raise CounterProtocolError(
            "drawlist_stats length does not match counters.drawlists"
        )

    aggregate_fields = {
        "vs_alu_instructions": sum(
            item.vertex.executed_alu_instructions for item in records
        ),
        "vs_tex_instructions": sum(
            item.vertex.executed_tex_instructions for item in records
        ),
        "vs_memory_instructions": sum(
            item.vertex.executed_memory_instructions for item in records
        ),
        "fs_alu_instructions": sum(
            item.fragment.executed_alu_instructions for item in records
        ),
        "fs_tex_instructions": sum(
            item.fragment.executed_tex_instructions for item in records
        ),
        "fs_memory_instructions": sum(
            item.fragment.executed_memory_instructions for item in records
        ),
    }
    for field, aggregate in aggregate_fields.items():
        if field in values and values[field] != aggregate:
            raise CounterProtocolError(
                f"drawlist_stats aggregate does not match counters.{field}"
            )
    return tuple(records)


def counter_record_from_message(message: Mapping[str, Any]) -> CounterRecord:
    if message.get("type") not in {"counter", "counter_sample"}:
        raise CounterProtocolError("Message is not a counter sample")
    raw_values = message.get("counters")
    if raw_values is None and isinstance(message.get("values"), list):
        raw_values = {
            item["id"]: item["value"]
            for item in message["values"]
            if isinstance(item, dict) and "id" in item and "value" in item
        }
    if not isinstance(raw_values, dict):
        raise CounterProtocolError("Counter sample has no value map")
    values = {str(key): _number(value) for key, value in raw_values.items()}
    drawlist_stats = _drawlist_records(message, values)
    return CounterRecord(
        frame=int(message.get("frame", 0)),
        marker=str(message.get("marker", message.get("test_id", ""))),
        values=values,
        source=str(message.get("source", "unknown")),
        provenance=str(message.get("provenance", "unknown")),
        virtual_time_ns=(
            int(message["virtual_time_ns"])
            if message.get("virtual_time_ns") is not None
            else None
        ),
        drawlist_stats=drawlist_stats,
    )


def human_number(value: int | float | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:,.3f}"
    return f"{int(value):,}"
