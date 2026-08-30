#!/usr/bin/env python3
"""Build a strict PvrGPU Mesa/RenderDoc POC command capsule.

The capsule binds one frozen RDC to both RenderDoc's structured OpenGL API
trace and Mesa's Gallium driver trace.  It intentionally supports only the
20 pinned GLBench cases; anything outside those exact draw signatures fails
closed before the SystemC model starts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Iterable, Sequence
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "config" / "rdc-glbench-v1.tsv"
SCHEMA = "pvrgpu.mesa-poc-command.v1"
MESA_DRIVER = "llvmpipe-gallium-trace-poc"


class CapsuleError(ValueError):
    """The replay evidence is not an accepted POC command."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(text: str) -> int:
    try:
        value = int(text, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected integer, got {text!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _load_manifest_case(manifest: Path, case_name: str) -> dict[str, str]:
    try:
        with manifest.open("r", encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source, delimiter="\t"))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise CapsuleError(f"cannot read manifest {manifest}: {exc}") from exc
    matches = [row for row in rows if row.get("case") == case_name]
    if len(matches) != 1:
        raise CapsuleError(
            f"manifest must contain exactly one {case_name!r} row, found {len(matches)}"
        )
    return matches[0]


def _expected_draw(case_name: str) -> tuple[str, str, int, bool]:
    if case_name.startswith("fill_solid") or case_name.startswith("fill_tex"):
        return ("glDrawArrays", "GL_TRIANGLE_STRIP", 4, False)
    if case_name.startswith("triangle_setup"):
        return ("glDrawElements", "GL_TRIANGLES", 98304, True)
    if case_name.startswith("attribute_fetch_shader"):
        return ("glDrawElements", "GL_TRIANGLES", 24576, True)
    if case_name.startswith("varyings_shader"):
        return ("glDrawElements", "GL_TRIANGLES", 96, True)
    raise CapsuleError(f"unsupported POC case: {case_name}")


def _expected_attribute_count(case_name: str) -> int:
    suffixes = {"_2_attr": 2, "_4_attr": 4, "_8_attr": 8}
    for suffix, count in suffixes.items():
        if case_name.endswith(suffix):
            return count
    return 1


def _expected_varying_count(case_name: str) -> int:
    match = re.fullmatch(r"varyings_shader_(1|2|4|8)", case_name)
    if not match:
        raise CapsuleError(f"invalid varying case name: {case_name}")
    return int(match.group(1))


def _extract_api_calls(api_text: str) -> tuple[str, int, list[str]]:
    source_match = re.search(r"^- Source: `([^`]+)`$", api_text, re.MULTILINE)
    count_match = re.search(r"^- API calls: ([0-9]+)$", api_text, re.MULTILINE)
    block_match = re.search(r"```text\n(.*?)\n```", api_text, re.DOTALL)
    if not source_match or not count_match or not block_match:
        raise CapsuleError("RenderDoc API Markdown is missing source/count/call block")
    calls = [line for line in block_match.group(1).splitlines() if line]
    declared_count = int(count_match.group(1))
    if declared_count != len(calls):
        raise CapsuleError(
            f"RenderDoc API call count mismatch: header={declared_count}, body={len(calls)}"
        )
    return source_match.group(1), declared_count, calls


def _shader_sources(calls: Iterable[str]) -> list[str]:
    shaders: list[str] = []
    pattern = re.compile(r"^glShaderSource .* sources=(\[.*\]), length=\[\]\)$")
    for call in calls:
        match = pattern.match(call)
        if not match:
            continue
        try:
            parts = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise CapsuleError(f"invalid glShaderSource source array: {exc}") from exc
        if not isinstance(parts, list) or not all(
            isinstance(part, str) for part in parts
        ):
            raise CapsuleError("glShaderSource sources must be an array of strings")
        shaders.append("".join(parts))
    if len(shaders) != 2:
        raise CapsuleError(
            f"POC requires exactly one vertex and one fragment shader, found {len(shaders)}"
        )
    return shaders


def _validate_shader_family(case_name: str, vertex: str, fragment: str) -> None:
    if case_name.startswith("fill_solid"):
        valid = (
            "attribute vec4 position" in vertex
            and "gl_Position = position" in vertex
            and "uniform vec4 color" in fragment
            and "gl_FragColor = color" in fragment
        )
    elif case_name.startswith("triangle_setup"):
        valid = (
            "attribute vec4 c;" in vertex
            and "gl_Position = c" in vertex
            and "uniform vec4 color" in fragment
        )
    elif case_name.startswith("attribute_fetch_shader"):
        expected = _expected_attribute_count(case_name)
        attributes = re.findall(r"attribute vec4 c([1-8]);", vertex)
        valid = (
            len(attributes) == expected
            and attributes == [str(index) for index in range(1, expected + 1)]
            and "gl_FragColor = vec4(0.5)" in fragment
        )
    elif case_name.startswith("varyings_shader"):
        expected = _expected_varying_count(case_name)
        vertex_varyings = set(re.findall(r"varying vec4 v([1-8]);", vertex))
        fragment_varyings = set(re.findall(r"varying vec4 v([1-8]);", fragment))
        names = {str(index) for index in range(1, expected + 1)}
        valid = (
            vertex_varyings == names
            and fragment_varyings == names
            and "gl_Position = c" in vertex
            and "gl_FragColor" in fragment
        )
    elif case_name.startswith("fill_tex"):
        valid = (
            "attribute vec4 position" in vertex
            and "attribute vec4 texcoord" in vertex
            and "uniform float scale" in vertex
            and "uniform sampler2D texture" in fragment
            and "texture2D(texture, v1.xy)" in fragment
        )
    else:
        valid = False
    if not valid:
        raise CapsuleError(f"shader sources do not match {case_name}")


def _validate_texture_state(case_name: str, calls: Sequence[str]) -> None:
    if not case_name.startswith("fill_tex"):
        return
    joined = "\n".join(calls)
    has_linear = "param=GL_LINEAR)" in joined
    has_trilinear = "param=GL_LINEAR_MIPMAP_LINEAR)" in joined
    if case_name == "fill_tex_nearest" and (has_linear or has_trilinear):
        raise CapsuleError("nearest texture case unexpectedly enables linear filtering")
    if case_name == "fill_tex_bilinear" and (not has_linear or has_trilinear):
        raise CapsuleError("bilinear texture case has the wrong filter sequence")
    if case_name.startswith("fill_tex_trilinear") and not has_trilinear:
        raise CapsuleError("trilinear texture case did not enable mip-linear filtering")


def _validate_api_draw(
    calls: Sequence[str], case_name: str
) -> tuple[str, int, bool]:
    expected_method, expected_primitive, expected_count, indexed = _expected_draw(
        case_name
    )
    draws = [call for call in calls if call.startswith("glDraw")]
    if len(draws) != 1:
        raise CapsuleError(f"POC requires exactly one API draw, found {len(draws)}")
    call = draws[0]
    if indexed:
        pattern = re.compile(
            r"^glDrawElements \(mode=([^,]+), count=([0-9]+), "
            r"type=([^,]+), indices=.*\)$"
        )
        match = pattern.match(call)
        if (
            not match
            or expected_method != "glDrawElements"
            or match.group(1) != expected_primitive
            or int(match.group(2)) != expected_count
            or match.group(3) != "GL_UNSIGNED_SHORT"
        ):
            raise CapsuleError(f"unexpected indexed draw for {case_name}: {call}")
    else:
        pattern = re.compile(
            r"^glDrawArrays \(mode=([^,]+), first=([0-9]+), count=([0-9]+)\)$"
        )
        match = pattern.match(call)
        if (
            not match
            or expected_method != "glDrawArrays"
            or match.group(1) != expected_primitive
            or int(match.group(2)) != 0
            or int(match.group(3)) != expected_count
        ):
            raise CapsuleError(f"unexpected array draw for {case_name}: {call}")
    primitive = "triangles" if indexed else "triangle_strip"
    return primitive, expected_count, indexed


def _member_uint(struct: ET.Element | None, name: str) -> int | None:
    if struct is None:
        return None
    for member in struct.findall("member"):
        if member.get("name") == name:
            value = member.find("uint")
            if value is not None and value.text is not None:
                try:
                    return int(value.text, 10)
                except ValueError:
                    return None
    return None


def _member_int(struct: ET.Element | None, name: str) -> int | None:
    if struct is None:
        return None
    for member in struct.findall("member"):
        if member.get("name") == name:
            value = member.find("int")
            if value is not None and value.text is not None:
                try:
                    return int(value.text, 10)
                except ValueError:
                    return None
    return None


def _member_text(
    struct: ET.Element | None, name: str, child_tag: str
) -> str | None:
    if struct is None:
        return None
    for member in struct.findall("member"):
        if member.get("name") == name:
            value = member.find(child_tag)
            if value is not None and value.text is not None:
                return value.text
    return None


def _arg_text(call: ET.Element, name: str, child_tag: str) -> str | None:
    argument = _arg(call, name)
    value = argument.find(child_tag) if argument is not None else None
    return value.text if value is not None else None


def _context_id(call: ET.Element) -> str | None:
    return _arg_text(call, "pipe", "ptr") or _arg_text(
        call, "context", "ptr"
    )


def _last_call(calls: Sequence[ET.Element], method: str) -> ET.Element | None:
    return next(
        (call for call in reversed(calls) if call.get("method") == method), None
    )


def _last_call_with_index(
    calls: Sequence[ET.Element], method: str
) -> tuple[int, ET.Element] | None:
    return next(
        (
            (index, calls[index])
            for index in range(len(calls) - 1, -1, -1)
            if calls[index].get("method") == method
        ),
        None,
    )


def _state_struct(call: ET.Element | None) -> ET.Element | None:
    state = _arg(call, "state") if call is not None else None
    return state.find("struct") if state is not None else None


def _bound_state_create_call(
    window: Sequence[ET.Element],
    *,
    create_method: str,
    bind_method: str,
    label: str,
) -> ET.Element:
    """Resolve the object actually bound at the end of a Gallium call window."""

    binding = _last_call_with_index(window, bind_method)
    if binding is None:
        raise CapsuleError(f"Gallium target draw has no {label} binding")
    bind_index, bind_call = binding
    state_id = _arg_text(bind_call, "state", "ptr")
    if state_id is None:
        raise CapsuleError(f"Gallium target draw has a null {label} binding")

    delete_method = create_method.replace("create_", "delete_", 1)
    create_call: ET.Element | None = None
    for call in window[:bind_index]:
        if (
            call.get("method") == create_method
            and _ret_text(call, "ptr") == state_id
        ):
            if create_call is not None:
                raise CapsuleError(
                    f"Gallium {label} handle was recreated while still live"
                )
            create_call = call
        elif (
            call.get("method") == delete_method
            and _arg_text(call, "state", "ptr") == state_id
        ):
            if create_call is None:
                raise CapsuleError(
                    f"Gallium {label} handle was deleted without a live object"
                )
            create_call = None
    if create_call is None:
        raise CapsuleError(
            f"Gallium target draw {label} binding has no prior creation record"
        )

    if any(
        (
            call.get("method") == delete_method
            and _arg_text(call, "state", "ptr") == state_id
        )
        or (
            call.get("method") == create_method
            and _ret_text(call, "ptr") == state_id
        )
        for call in window[bind_index + 1 :]
    ):
        raise CapsuleError(
            f"Gallium target draw uses a deleted {label} binding"
        )
    return create_call


def _live_created_object_call(
    window: Sequence[ET.Element],
    *,
    create_method: str,
    delete_method: str,
    delete_arg: str,
    object_id: str,
    label: str,
) -> ET.Element:
    live: ET.Element | None = None
    for call in window:
        if (
            call.get("method") == create_method
            and _ret_text(call, "ptr") == object_id
        ):
            if live is not None:
                raise CapsuleError(
                    f"Gallium {label} handle was recreated while still live"
                )
            live = call
        elif (
            call.get("method") == delete_method
            and _arg_text(call, delete_arg, "ptr") == object_id
        ):
            if live is None:
                raise CapsuleError(
                    f"Gallium {label} handle was deleted without a live object"
                )
            live = None
    if live is None:
        raise CapsuleError(f"Gallium target draw has no live {label} object")
    return live


def _pointer_array(argument: ET.Element | None, label: str) -> list[str | None]:
    array = argument.find("array") if argument is not None else None
    if array is None:
        raise CapsuleError(f"Gallium {label} is not a pointer array")
    pointers: list[str | None] = []
    for element in array.findall("elem"):
        pointer = element.find("ptr")
        null = element.find("null")
        if pointer is not None and pointer.text and null is None:
            pointers.append(pointer.text)
        elif null is not None and pointer is None:
            pointers.append(None)
        else:
            raise CapsuleError(f"Gallium {label} has a malformed pointer slot")
    return pointers


def _current_constant_hex(
    window: Sequence[ET.Element], shader: str
) -> str:
    current = next(
        (
            call
            for call in reversed(window)
            if call.get("method") == "set_constant_buffer"
            and _arg_text(call, "shader", "enum") == shader
            and _arg_text(call, "index", "uint") == "0"
        ),
        None,
    )
    if current is None:
        return "NONE"

    constant_arg = _arg(current, "constant_buffer")
    if constant_arg is None:
        raise CapsuleError(f"target draw has malformed {shader} constant binding")
    if constant_arg.find("null") is not None:
        if _arg_text(current, "data", "bytes") is not None:
            raise CapsuleError(
                f"target draw has bytes for a null {shader} constant binding"
            )
        return "NONE"

    constant = constant_arg.find("struct")
    data = _arg_text(current, "data", "bytes")
    size = _member_uint(constant, "buffer_size")
    if (
        constant is None
        or _member_text(constant, "buffer", "ptr") is None
        or size is None
        or size <= 0
        or data is None
        or len(data) % 2 != 0
        or not re.fullmatch(r"[0-9A-Fa-f]+", data)
        or len(data) != size * 2
    ):
        raise CapsuleError(
            f"target draw has incomplete {shader} constant bytes"
        )
    return data.upper()


def _bool_text(value: str | None, field: str) -> int:
    if value not in {"0", "1"}:
        raise CapsuleError(f"Gallium {field} is not a boolean")
    return int(value)


def _frame1_command_inputs(
    window: Sequence[ET.Element], *, width: int, height: int
) -> dict[str, str | int]:
    """Extract the actual bytes/state which drive the target POC draw."""

    framebuffer_binding = _last_call_with_index(window, "set_framebuffer_state")
    if framebuffer_binding is None:
        framebuffer_binding = _last_call_with_index(
            window, "current_framebuffer_state"
        )
    if framebuffer_binding is None:
        raise CapsuleError("Gallium target draw has no framebuffer binding")
    framebuffer_index, framebuffer_call = framebuffer_binding
    framebuffer = _state_struct(framebuffer_call)
    if (
        _member_uint(framebuffer, "width") != width
        or _member_uint(framebuffer, "height") != height
    ):
        raise CapsuleError(
            "Gallium target draw framebuffer does not match runner dimensions"
        )
    cbufs_member = next(
        (
            member
            for member in framebuffer.findall("member")
            if member.get("name") == "cbufs"
        ),
        None,
    ) if framebuffer is not None else None
    cbuf = (
        cbufs_member.find("./array/elem/struct")
        if cbufs_member is not None
        else None
    )
    framebuffer_format = _member_text(cbuf, "format", "enum")

    dsa = _state_struct(
        _bound_state_create_call(
            window,
            create_method="create_depth_stencil_alpha_state",
            bind_method="bind_depth_stencil_alpha_state",
            label="depth/stencil/alpha state",
        )
    )
    blend = _state_struct(
        _bound_state_create_call(
            window,
            create_method="create_blend_state",
            bind_method="bind_blend_state",
            label="blend state",
        )
    )
    raster = _state_struct(
        _bound_state_create_call(
            window,
            create_method="create_rasterizer_state",
            bind_method="bind_rasterizer_state",
            label="rasterizer state",
        )
    )
    if dsa is None or blend is None or raster is None:
        raise CapsuleError("Gallium target draw is missing depth/blend/raster state")

    rt_member = next(
        (member for member in blend.findall("member") if member.get("name") == "rt"),
        None,
    )
    rt = rt_member.find("./array/elem/struct") if rt_member is not None else None
    if rt is None:
        raise CapsuleError("Gallium target draw is missing render-target blend state")

    vertex_elements_call = _bound_state_create_call(
        window,
        create_method="create_vertex_elements_state",
        bind_method="bind_vertex_elements_state",
        label="vertex-elements state",
    )
    elements_arg = _arg(vertex_elements_call, "elements") if vertex_elements_call is not None else None
    elements = (
        elements_arg.findall("./array/elem/struct")
        if elements_arg is not None
        else []
    )
    num_elements_text = (
        _arg_text(vertex_elements_call, "num_elements", "uint")
        if vertex_elements_call is not None
        else None
    )
    if not elements or num_elements_text is None:
        raise CapsuleError("Gallium target draw is missing vertex element state")
    try:
        num_elements = int(num_elements_text, 10)
    except ValueError as exc:
        raise CapsuleError("Gallium vertex element count is invalid") from exc
    if num_elements <= 0 or len(elements) != num_elements:
        raise CapsuleError(
            "Gallium vertex element array does not match its declared count"
        )

    vertex_buffer_call = _last_call(window, "set_vertex_buffers")
    num_buffers_text = (
        _arg_text(vertex_buffer_call, "num_buffers", "uint")
        if vertex_buffer_call is not None
        else None
    )
    buffers_arg = (
        _arg(vertex_buffer_call, "buffers")
        if vertex_buffer_call is not None
        else None
    )
    buffer_structs = (
        buffers_arg.findall("./array/elem/struct")
        if buffers_arg is not None
        else []
    )
    buffer_data_arg = (
        _arg(vertex_buffer_call, "buffer_data")
        if vertex_buffer_call is not None
        else None
    )
    vertex_buffers = (
        [value.text for value in buffer_data_arg.findall("./array/elem/bytes") if value.text]
        if buffer_data_arg is not None
        else []
    )
    try:
        num_buffers = int(num_buffers_text, 10) if num_buffers_text is not None else 0
    except ValueError as exc:
        raise CapsuleError("Gallium vertex buffer count is invalid") from exc
    if (
        num_buffers <= 0
        or len(buffer_structs) != num_buffers
        or len(vertex_buffers) != num_buffers
    ):
        raise CapsuleError(
            "Gallium vertex buffer arrays do not match their declared count"
        )
    if any(
        len(data) % 2 != 0 or not re.fullmatch(r"[0-9A-Fa-f]+", data)
        for data in vertex_buffers
    ):
        raise CapsuleError("Gallium vertex buffer has invalid byte encoding")

    buffer_indices = [
        _member_uint(element, "vertex_buffer_index") for element in elements
    ]
    if buffer_indices != list(range(num_buffers)):
        raise CapsuleError(
            "Gallium vertex elements must map one-to-one to contiguous buffer slots"
        )

    canonical_element = elements[0]
    element_fields = (
        "src_format",
        "src_stride",
        "src_offset",
        "instance_divisor",
        "dual_slot",
    )
    canonical_values = {
        "src_format": _member_text(canonical_element, "src_format", "enum"),
        "src_stride": _member_uint(canonical_element, "src_stride"),
        "src_offset": _member_uint(canonical_element, "src_offset"),
        "instance_divisor": _member_uint(canonical_element, "instance_divisor"),
        "dual_slot": _member_text(canonical_element, "dual_slot", "bool"),
    }
    if any(value is None for value in canonical_values.values()):
        raise CapsuleError("Gallium vertex element is missing required fields")
    for element in elements[1:]:
        values = {
            "src_format": _member_text(element, "src_format", "enum"),
            "src_stride": _member_uint(element, "src_stride"),
            "src_offset": _member_uint(element, "src_offset"),
            "instance_divisor": _member_uint(element, "instance_divisor"),
            "dual_slot": _member_text(element, "dual_slot", "bool"),
        }
        if any(values[field] != canonical_values[field] for field in element_fields):
            raise CapsuleError(
                "Gallium POC requires byte-identical vertex element layouts"
            )
    if (
        canonical_values["instance_divisor"] != 0
        or canonical_values["dual_slot"] != "0"
    ):
        raise CapsuleError("Gallium POC does not support instanced or dual-slot elements")

    resource_pointers = [
        _member_text(buffer, "buffer.resource", "ptr") for buffer in buffer_structs
    ]
    if (
        any(
            _member_text(buffer, "is_user_buffer", "bool") != "0"
            for buffer in buffer_structs
        )
        or any(_member_uint(buffer, "buffer_offset") != 0 for buffer in buffer_structs)
        or any(pointer is None for pointer in resource_pointers)
    ):
        raise CapsuleError(
            "Gallium POC requires resource-backed vertex buffers at offset zero"
        )

    canonical_buffers: list[str] = []
    canonical_pointers: list[str] = []
    vertex_buffer_map: list[int] = []
    for pointer, data in zip(resource_pointers, vertex_buffers, strict=True):
        if pointer in canonical_pointers:
            canonical_index = canonical_pointers.index(pointer)
            if canonical_buffers[canonical_index] != data:
                raise CapsuleError(
                    "one Gallium vertex resource produced inconsistent bytes"
                )
        else:
            canonical_index = len(canonical_buffers)
            canonical_pointers.append(pointer)
            canonical_buffers.append(data)
        vertex_buffer_map.append(canonical_index)
    if len(canonical_buffers) > 2:
        raise CapsuleError("Gallium POC supports at most two distinct vertex buffers")

    vertex_constant_hex = _current_constant_hex(window, "MESA_SHADER_VERTEX")
    fragment_constant_hex = _current_constant_hex(window, "MESA_SHADER_FRAGMENT")

    clear: ET.Element | None = None
    for call in window[framebuffer_index + 1 :]:
        if call.get("method") != "clear":
            continue
        buffers_text = _arg_text(call, "buffers", "uint")
        color_mask_text = _arg_text(call, "color_clear_mask", "uint")
        try:
            buffers = int(buffers_text, 10) if buffers_text is not None else -1
            color_mask = (
                int(color_mask_text, 10)
                if color_mask_text is not None
                else -1
            )
        except ValueError as exc:
            raise CapsuleError("Gallium clear masks are malformed") from exc
        if buffers < 0 or color_mask < 0:
            raise CapsuleError("Gallium clear masks are missing")
        if buffers & 0x4 and color_mask & 0x1:
            clear = call
    clear_arg = _arg(clear, "color->ui") if clear is not None else None
    clear_values = (
        [value.text for value in clear_arg.findall("./array/elem/uint") if value.text]
        if clear_arg is not None
        else []
    )
    if len(clear_values) != 4:
        raise CapsuleError("Gallium target draw requires one RGBA clear value")

    required_text = {
        "framebuffer_format": framebuffer_format,
        "blend_rgb_func": _member_text(rt, "rgb_func", "enum"),
        "blend_rgb_src_factor": _member_text(rt, "rgb_src_factor", "enum"),
        "blend_rgb_dst_factor": _member_text(rt, "rgb_dst_factor", "enum"),
        "blend_alpha_func": _member_text(rt, "alpha_func", "enum"),
        "blend_alpha_src_factor": _member_text(
            rt, "alpha_src_factor", "enum"
        ),
        "blend_alpha_dst_factor": _member_text(
            rt, "alpha_dst_factor", "enum"
        ),
        "vertex_format": canonical_values["src_format"],
    }
    missing = [name for name, value in required_text.items() if value is None]
    if missing:
        raise CapsuleError(
            "Gallium target draw is missing state fields: " + ", ".join(missing)
        )

    return {
        **{name: value for name, value in required_text.items() if value is not None},
        "clear_color_bits": ",".join(clear_values),
        "depth_enabled": _bool_text(
            _member_text(dsa, "depth_enabled", "bool"), "depth_enabled"
        ),
        "depth_write": _bool_text(
            _member_text(dsa, "depth_writemask", "bool"), "depth_writemask"
        ),
        "depth_func": _member_uint(dsa, "depth_func")
        if _member_uint(dsa, "depth_func") is not None
        else -1,
        "blend_enabled": _member_uint(rt, "blend_enable")
        if _member_uint(rt, "blend_enable") is not None
        else -1,
        "front_ccw": _member_uint(raster, "front_ccw")
        if _member_uint(raster, "front_ccw") is not None
        else -1,
        "cull_face": _member_uint(raster, "cull_face")
        if _member_uint(raster, "cull_face") is not None
        else -1,
        "vertex_elements": num_elements,
        "vertex_stride": int(canonical_values["src_stride"]),
        "vertex_offset": int(canonical_values["src_offset"]),
        "vertex_buffer_map": ",".join(str(index) for index in vertex_buffer_map),
        "vertex_buffer_hex": canonical_buffers[0].upper(),
        "vertex_buffer_1_hex": (
            canonical_buffers[1].upper() if len(canonical_buffers) == 2 else "NONE"
        ),
        "vertex_constant_hex": vertex_constant_hex,
        "fragment_constant_hex": fragment_constant_hex,
    }


def _member_struct(struct: ET.Element | None, name: str) -> ET.Element | None:
    if struct is None:
        return None
    member = next(
        (item for item in struct.findall("member") if item.get("name") == name),
        None,
    )
    return member.find("struct") if member is not None else None


def _ret_text(call: ET.Element, child_tag: str) -> str | None:
    value = call.find(f"./ret/{child_tag}")
    return value.text if value is not None else None


def _no_texture_inputs() -> dict[str, str | int]:
    return {
        "texture_format": "NONE",
        "texture_width": 0,
        "texture_height": 0,
        "texture_levels": 0,
        "texture_mip_sizes": "NONE",
        "texture_mip_hex": "NONE",
        "sampler_min_img_filter": 0,
        "sampler_min_mip_filter": 0,
        "sampler_mag_img_filter": 0,
    }


def _texture_command_inputs(
    calls: Sequence[ET.Element],
    draw_index: int,
    *,
    case_name: str,
    width: int,
    height: int,
) -> dict[str, str | int]:
    """Resolve the sampled RGBA mip chain and filter state for one draw."""

    prefix = calls[:draw_index]
    draw_pipe = _context_id(calls[draw_index])
    if draw_pipe is None:
        raise CapsuleError("texture target draw has no pipe context")
    sampler_views_binding = next(
        (
            (index, prefix[index])
            for index in range(len(prefix) - 1, -1, -1)
            if prefix[index].get("method") == "set_sampler_views"
            and _context_id(prefix[index]) == draw_pipe
            and _arg_text(prefix[index], "shader", "enum")
            == "MESA_SHADER_FRAGMENT"
            and _arg_text(prefix[index], "start", "uint") == "0"
        ),
        None,
    )
    sampler_views_index, sampler_views_call = (
        sampler_views_binding if sampler_views_binding is not None else (-1, None)
    )
    view_ids = _pointer_array(
        _arg(sampler_views_call, "views")
        if sampler_views_call is not None
        else None,
        "sampler-view slots",
    )
    if len(view_ids) != 1 or view_ids[0] is None:
        raise CapsuleError("texture target draw requires one fragment sampler view")
    view_id = view_ids[0]
    context_before_view_bind = [
        call
        for call in prefix[:sampler_views_index]
        if _context_id(call) == draw_pipe
    ]
    create_view = _live_created_object_call(
        context_before_view_bind,
        create_method="create_sampler_view",
        delete_method="sampler_view_destroy",
        delete_arg="view",
        object_id=view_id,
        label="sampler-view",
    )
    create_view_index = next(
        index for index, call in enumerate(prefix) if call is create_view
    )
    context_before_draw = [
        call for call in prefix if _context_id(call) == draw_pipe
    ]
    if (
        _live_created_object_call(
            context_before_draw,
            create_method="create_sampler_view",
            delete_method="sampler_view_destroy",
            delete_arg="view",
            object_id=view_id,
            label="sampler-view",
        )
        is not create_view
    ):
        raise CapsuleError("texture target sampler view changed generation after bind")
    resource_id = _arg_text(create_view, "resource", "ptr")
    templ_arg = _arg(create_view, "templ")
    view = templ_arg.find("struct") if templ_arg is not None else None
    view_u = _member_struct(view, "u")
    view_tex = _member_struct(view_u, "tex")
    if (
        resource_id is None
        or _member_text(view, "format", "enum")
        != "PIPE_FORMAT_R8G8B8A8_UNORM"
        or _member_text(view, "target", "enum") != "PIPE_TEXTURE_2D"
        or _member_uint(view_tex, "first_layer") != 0
        or _member_uint(view_tex, "last_layer") != 0
        or _member_uint(view_tex, "first_level") != 0
        or [_member_uint(view, f"swizzle_{name}") for name in "rgba"]
        != [0, 1, 2, 3]
    ):
        raise CapsuleError("draw sampler view is not the required RGBA8 2D view")
    last_level = _member_uint(view_tex, "last_level")
    if last_level is None:
        raise CapsuleError("texture sampler view has no last mip level")

    resource_creates = [
        call
        for call in prefix[:create_view_index]
        if call.get("method") == "resource_create"
        and _ret_text(call, "ptr") == resource_id
    ]
    if len(resource_creates) != 1:
        raise CapsuleError(
            "sampled texture resource has an ambiguous creation generation"
        )
    resource_create = resource_creates[0]
    resource_template = (
        _arg(resource_create, "templat") if resource_create is not None else None
    )
    resource = (
        resource_template.find("struct")
        if resource_template is not None
        else None
    )
    levels = last_level + 1
    if (
        resource is None
        or _member_text(resource, "target", "enum") != "PIPE_TEXTURE_2D"
        or _member_text(resource, "format", "enum")
        != "PIPE_FORMAT_R8G8B8A8_UNORM"
        or _member_uint(resource, "width") != width
        or _member_uint(resource, "height") != height
        or _member_uint(resource, "depth") != 1
        or _member_uint(resource, "array_size") != 1
        or _member_uint(resource, "last_level") != last_level
        or levels != 10
    ):
        raise CapsuleError("sampled texture descriptor is outside the GLBench gate")

    bind_sampler_with_index = next(
        (
            (index, prefix[index])
            for index in range(len(prefix) - 1, -1, -1)
            if prefix[index].get("method") == "bind_sampler_states"
            and _context_id(prefix[index]) == draw_pipe
            and _arg_text(prefix[index], "shader", "enum")
            == "MESA_SHADER_FRAGMENT"
            and _arg_text(prefix[index], "start", "uint") == "0"
        ),
        None,
    )
    bind_sampler_index, bind_sampler = (
        bind_sampler_with_index if bind_sampler_with_index is not None else (-1, None)
    )
    state_ids = _pointer_array(
        _arg(bind_sampler, "states") if bind_sampler is not None else None,
        "sampler-state slots",
    )
    if len(state_ids) != 1 or state_ids[0] is None:
        raise CapsuleError("texture target draw requires one fragment sampler state")
    if _arg_text(bind_sampler, "num_states", "uint") != "1":
        raise CapsuleError("texture target sampler bind must contain exactly one state")
    sampler_id = state_ids[0]
    context_before_sampler_bind = [
        call
        for call in prefix[:bind_sampler_index]
        if _context_id(call) == draw_pipe
    ]
    create_sampler = _live_created_object_call(
        context_before_sampler_bind,
        create_method="create_sampler_state",
        delete_method="delete_sampler_state",
        delete_arg="state",
        object_id=sampler_id,
        label="sampler-state",
    )
    if (
        _live_created_object_call(
            context_before_draw,
            create_method="create_sampler_state",
            delete_method="delete_sampler_state",
            delete_arg="state",
            object_id=sampler_id,
            label="sampler-state",
        )
        is not create_sampler
    ):
        raise CapsuleError("texture target sampler state changed generation after bind")
    state = _state_struct(create_sampler)
    border_member = next(
        (
            member
            for member in state.findall("member")
            if member.get("name") == "border_color.f"
        ),
        None,
    ) if state is not None else None
    border_values = (
        [item.text for item in border_member.findall("./array/elem/float")]
        if border_member is not None
        else []
    )
    min_img = _member_uint(state, "min_img_filter")
    min_mip = _member_uint(state, "min_mip_filter")
    mag_img = _member_uint(state, "mag_img_filter")
    if case_name == "fill_tex_nearest":
        expected_filters = (0, 2, 0)
    elif case_name == "fill_tex_bilinear":
        expected_filters = (1, 2, 1)
    else:
        expected_filters = (1, 1, 1)
    if (
        state is None
        or (min_img, min_mip, mag_img) != expected_filters
        or any(_member_uint(state, name) != 0 for name in ("wrap_s", "wrap_t", "wrap_r"))
        or _member_uint(state, "compare_mode") != 0
        or _member_uint(state, "compare_func") != 3
        or _member_text(state, "unnormalized_coords", "bool") != "0"
        or _member_uint(state, "max_anisotropy") != 0
        or _member_text(state, "seamless_cube_map", "bool") != "1"
        or _member_text(state, "lod_bias", "float") != "0"
        or _member_text(state, "min_lod", "float") != "0"
        or _member_text(state, "max_lod", "float") != "1000"
        or border_values != ["0", "0", "0", "0"]
        or _member_text(state, "border_color_format", "enum")
        != "PIPE_FORMAT_NONE"
    ):
        raise CapsuleError("draw sampler state does not match the API filter state")

    resource_descriptors: dict[str, ET.Element] = {}
    contents: dict[tuple[str, int], bytes | None] = {}
    for call in prefix:
        method = call.get("method")
        if method == "resource_create":
            pointer = _ret_text(call, "ptr")
            templat = _arg(call, "templat")
            descriptor = templat.find("struct") if templat is not None else None
            if pointer is not None and descriptor is not None:
                if pointer in resource_descriptors:
                    raise CapsuleError(
                        "Gallium resource pointer was recreated without "
                        "traceable destruction"
                    )
                resource_descriptors[pointer] = descriptor
            continue
        if method == "resource_destroy":
            pointer = _arg_text(call, "resource", "ptr")
            if pointer is not None:
                resource_descriptors.pop(pointer, None)
                contents = {
                    key: value
                    for key, value in contents.items()
                    if key[0] != pointer
                }
            continue
        if method == "texture_subdata":
            pointer = _arg_text(call, "resource", "ptr")
            level_text = _arg_text(call, "level", "uint")
            if pointer is None or level_text is None:
                raise CapsuleError("Gallium texture write has no resource or level")
            try:
                level = int(level_text, 10)
            except ValueError as exc:
                raise CapsuleError("Gallium texture write level is malformed") from exc
            descriptor = resource_descriptors.get(pointer)
            if descriptor is None:
                raise CapsuleError("Gallium texture write targets an unknown resource")
            if (
                _member_text(descriptor, "format", "enum")
                != "PIPE_FORMAT_R8G8B8A8_UNORM"
            ):
                continue
            key = (pointer, level)
            descriptor_width = _member_uint(descriptor, "width")
            descriptor_height = _member_uint(descriptor, "height")
            descriptor_last_level = _member_uint(descriptor, "last_level")
            if (
                descriptor_width is None
                or descriptor_height is None
                or descriptor_last_level is None
                or level < 0
                or level > descriptor_last_level
            ):
                contents[key] = None
                continue
            level_width = max(1, descriptor_width >> level)
            level_height = max(1, descriptor_height >> level)
            box_arg = _arg(call, "box")
            box = box_arg.find("struct") if box_arg is not None else None
            data = _arg_text(call, "data", "bytes")
            expected_size = level_width * level_height * 4
            if (
                box is None
                or _member_int(box, "x") != 0
                or _member_int(box, "y") != 0
                or _member_int(box, "z") != 0
                or _member_int(box, "width") != level_width
                or _member_int(box, "height") != level_height
                or _member_int(box, "depth") != 1
                or _arg_text(call, "stride", "uint") != str(level_width * 4)
                or _arg_text(call, "layer_stride", "uint")
                != str(expected_size)
                or data is None
                or not re.fullmatch(r"[0-9A-Fa-f]+", data)
                or len(data) != expected_size * 2
            ):
                contents[key] = None
                continue
            contents[key] = bytes.fromhex(data)
        elif method == "resource_copy_region":
            source = _arg_text(call, "src", "ptr")
            destination = _arg_text(call, "dst", "ptr")
            source_level_text = _arg_text(call, "src_level", "uint")
            destination_level_text = _arg_text(call, "dst_level", "uint")
            if destination is None or destination_level_text is None:
                raise CapsuleError("Gallium resource copy has no destination identity")
            try:
                destination_level = int(destination_level_text, 10)
            except ValueError as exc:
                raise CapsuleError(
                    "Gallium resource copy destination level is malformed"
                ) from exc
            destination_descriptor = resource_descriptors.get(destination)
            if destination_descriptor is None:
                raise CapsuleError(
                    "Gallium resource copy targets an unknown resource"
                )
            if (
                _member_text(destination_descriptor, "format", "enum")
                != "PIPE_FORMAT_R8G8B8A8_UNORM"
            ):
                continue
            destination_key = (destination, destination_level)
            if source is None or source_level_text is None:
                contents[destination_key] = None
                continue
            try:
                source_level = int(source_level_text, 10)
            except ValueError:
                contents[destination_key] = None
                continue
            source_bytes = contents.get((source, source_level))
            descriptor_width = _member_uint(destination_descriptor, "width")
            descriptor_height = _member_uint(destination_descriptor, "height")
            descriptor_last_level = _member_uint(
                destination_descriptor, "last_level"
            )
            if (
                descriptor_width is None
                or descriptor_height is None
                or descriptor_last_level is None
                or destination_level < 0
                or destination_level > descriptor_last_level
            ):
                contents[destination_key] = None
                continue
            destination_width = max(1, descriptor_width >> destination_level)
            destination_height = max(1, descriptor_height >> destination_level)
            source_box_arg = _arg(call, "src_box")
            source_box = (
                source_box_arg.find("struct")
                if source_box_arg is not None
                else None
            )
            if (
                _arg_text(call, "dstx", "uint") != "0"
                or _arg_text(call, "dsty", "uint") != "0"
                or _arg_text(call, "dstz", "uint") != "0"
                or _member_int(source_box, "x") != 0
                or _member_int(source_box, "y") != 0
                or _member_int(source_box, "z") != 0
                or _member_int(source_box, "width") != destination_width
                or _member_int(source_box, "height") != destination_height
                or _member_int(source_box, "depth") != 1
                or source_bytes is None
                or len(source_bytes) != destination_width * destination_height * 4
            ):
                contents[destination_key] = None
                continue
            contents[destination_key] = source_bytes

    if resource_descriptors.get(resource_id) is not resource:
        raise CapsuleError("sampled texture resource was destroyed or pointer-reused")

    mip_bytes: list[bytes] = []
    mip_sizes: list[int] = []
    for level in range(levels):
        data = contents.get((resource_id, level))
        expected_size = max(1, width >> level) * max(1, height >> level) * 4
        if data is None or len(data) != expected_size:
            raise CapsuleError(f"sampled texture mip {level} has no exact upload bytes")
        mip_bytes.append(data)
        mip_sizes.append(len(data))
    texture_bytes = b"".join(mip_bytes)
    if len(texture_bytes) > 16 * 1024 * 1024:
        raise CapsuleError("sampled texture mip chain exceeds the POC bound")
    return {
        "texture_format": "PIPE_FORMAT_R8G8B8A8_UNORM",
        "texture_width": width,
        "texture_height": height,
        "texture_levels": levels,
        "texture_mip_sizes": ",".join(str(size) for size in mip_sizes),
        "texture_mip_hex": texture_bytes.hex().upper(),
        "sampler_min_img_filter": int(min_img),
        "sampler_min_mip_filter": int(min_mip),
        "sampler_mag_img_filter": int(mag_img),
    }


def _arg(call: ET.Element, name: str) -> ET.Element | None:
    return next((arg for arg in call.findall("arg") if arg.get("name") == name), None)


def _gallium_evidence(
    trace_path: Path,
    *,
    case_name: str,
    width: int,
    height: int,
    draw_count: int,
    indexed: bool,
) -> dict[str, str | int]:
    try:
        tree = ET.parse(trace_path)
    except (OSError, ET.ParseError) as exc:
        raise CapsuleError(f"cannot parse Gallium trace {trace_path}: {exc}") from exc
    root = tree.getroot()
    if root.tag != "trace" or root.get("version") != "0.1":
        raise CapsuleError("unsupported Gallium trace schema")

    calls = root.findall("call")
    gallium_draws = 0
    target_draws = 0
    framebuffer_matches = 0
    resource_creates = 0
    state_calls = 0
    constant_bytes = 0
    expected_mode = 4 if indexed else 5
    expected_index_size = 2 if indexed else 0
    state_methods = {
        "set_framebuffer_state",
        "current_framebuffer_state",
        "create_depth_stencil_alpha_state",
        "bind_depth_stencil_alpha_state",
        "create_blend_state",
        "bind_blend_state",
        "create_rasterizer_state",
        "bind_rasterizer_state",
        "set_viewport_states",
        "set_scissor_states",
        "set_constant_buffer",
        "create_vertex_elements_state",
        "bind_vertex_elements_state",
        "set_vertex_buffers",
        "set_sampler_views",
        "bind_sampler_states",
    }

    matching_draw_indices: list[int] = []
    for call_index, call in enumerate(calls):
        method = call.get("method", "")
        if method == "resource_create":
            resource_creates += 1
        if method in state_methods:
            state_calls += 1
        if method in {"set_framebuffer_state", "current_framebuffer_state"}:
            state_arg = _arg(call, "state")
            state_struct = state_arg.find("struct") if state_arg is not None else None
            if (
                _member_uint(state_struct, "width") == width
                and _member_uint(state_struct, "height") == height
            ):
                framebuffer_matches += 1
        if method == "set_constant_buffer":
            data_arg = _arg(call, "data")
            data = data_arg.find("bytes") if data_arg is not None else None
            if data is not None and data.text:
                if len(data.text) % 2 or not re.fullmatch(r"[0-9A-Fa-f]+", data.text):
                    raise CapsuleError("Gallium constant buffer has invalid byte encoding")
                constant_bytes += len(data.text) // 2
        if method != "draw_vbo":
            continue
        gallium_draws += 1
        info_arg = _arg(call, "info")
        info = info_arg.find("struct") if info_arg is not None else None
        draws_arg = _arg(call, "draws")
        counts: list[int] = []
        if draws_arg is not None:
            for draw_struct in draws_arg.findall("./array/elem/struct"):
                count = _member_uint(draw_struct, "count")
                if count is not None:
                    counts.append(count)
        if (
            _member_uint(info, "mode") == expected_mode
            and _member_uint(info, "index_size") == expected_index_size
            and sum(counts) == draw_count
        ):
            target_draws += 1
            matching_draw_indices.append(call_index)

    if gallium_draws == 0 or target_draws == 0:
        raise CapsuleError("Gallium trace contains no matching target draw_vbo")
    if framebuffer_matches == 0:
        raise CapsuleError("Gallium trace contains no matching framebuffer state")
    if resource_creates == 0 or state_calls == 0:
        raise CapsuleError("Gallium trace contains no resource/state command evidence")

    texture_inputs = _no_texture_inputs()
    if case_name.startswith("fill_tex"):
        selected_draws: list[int] = []
        selected_textures: list[dict[str, str | int]] = []
        candidate_errors: list[str] = []
        for draw_index in matching_draw_indices:
            try:
                candidate = _texture_command_inputs(
                    calls,
                    draw_index,
                    case_name=case_name,
                    width=width,
                    height=height,
                )
            except CapsuleError as exc:
                candidate_errors.append(str(exc))
                continue
            selected_draws.append(draw_index)
            selected_textures.append(candidate)
        if not selected_draws:
            detail = candidate_errors[-1] if candidate_errors else "no candidate"
            raise CapsuleError(f"Gallium trace has no exact texture target: {detail}")
        texture_inputs = selected_textures[0]
        if any(item != texture_inputs for item in selected_textures[1:]):
            raise CapsuleError(
                "matching Gallium texture targets do not have identical resources"
            )
        matching_draw_indices = selected_draws
        target_draws = len(selected_draws)

    command_snapshots = []
    for draw_index in matching_draw_indices:
        draw_pipe = _context_id(calls[draw_index])
        if draw_pipe is None:
            raise CapsuleError("Gallium target draw has no pipe context")
        context_window = [
            call
            for call in calls[:draw_index]
            if _context_id(call) == draw_pipe
        ]
        command_snapshots.append(
            _frame1_command_inputs(
                context_window, width=width, height=height
            )
        )
    command_inputs = command_snapshots[0]
    if any(snapshot != command_inputs for snapshot in command_snapshots[1:]):
        raise CapsuleError(
            "matching Gallium target draws do not have identical command state"
        )

    index_buffers: list[str] = []
    for draw_index in matching_draw_indices:
        index_arg = _arg(calls[draw_index], "index_data")
        if indexed:
            index_data = index_arg.find("bytes") if index_arg is not None else None
            if (
                index_data is None
                or index_data.text is None
                or not re.fullmatch(r"[0-9A-Fa-f]+", index_data.text)
                or len(index_data.text) != draw_count * expected_index_size * 2
            ):
                raise CapsuleError(
                    "Gallium indexed target draw is missing its exact index bytes"
                )
            index_buffers.append(index_data.text.upper())
        else:
            if index_arg is None or index_arg.find("null") is None:
                raise CapsuleError("Gallium non-indexed target draw has index bytes")
            index_buffers.append("NONE")
    index_buffer_hex = index_buffers[0]
    if any(data != index_buffer_hex for data in index_buffers[1:]):
        raise CapsuleError(
            "matching Gallium target draws do not have identical index bytes"
        )
    return {
        "gallium_draw_calls": gallium_draws,
        "gallium_target_draws": target_draws,
        "gallium_framebuffer_matches": framebuffer_matches,
        "gallium_resource_creates": resource_creates,
        "gallium_state_calls": state_calls,
        "gallium_constant_bytes": constant_bytes,
        "index_size": expected_index_size,
        "index_buffer_hex": index_buffer_hex,
        **texture_inputs,
        **command_inputs,
    }


def build_capsule(
    *,
    rdc: Path,
    api_trace: Path,
    gallium_trace: Path,
    manifest: Path,
    case_name: str,
    width: int,
    height: int,
) -> dict[str, str | int]:
    for path, label in (
        (rdc, "RDC"),
        (api_trace, "RenderDoc API trace"),
        (gallium_trace, "Gallium trace"),
        (manifest, "manifest"),
    ):
        if not path.is_file():
            raise CapsuleError(f"{label} is missing: {path}")

    row = _load_manifest_case(manifest, case_name)
    if int(row["width"]) != width or int(row["height"]) != height:
        raise CapsuleError("runner dimensions do not match the frozen manifest")
    rdc_sha256 = _sha256(rdc)
    if row["rdc_sha256"] != rdc_sha256:
        raise CapsuleError("RDC SHA-256 does not match the frozen manifest")
    if Path(row["rdc_path"]).name != rdc.name:
        raise CapsuleError("RDC filename does not match the frozen manifest")

    try:
        api_text = api_trace.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise CapsuleError(f"cannot read RenderDoc API trace: {exc}") from exc
    source, api_calls, calls = _extract_api_calls(api_text)
    if Path(source).resolve() != rdc.resolve():
        raise CapsuleError("RenderDoc API trace source is not the requested RDC")
    primitive, draw_count, indexed = _validate_api_draw(calls, case_name)
    vertex_shader, fragment_shader = _shader_sources(calls)
    _validate_shader_family(case_name, vertex_shader, fragment_shader)
    _validate_texture_state(case_name, calls)
    gallium = _gallium_evidence(
        gallium_trace,
        case_name=case_name,
        width=width,
        height=height,
        draw_count=draw_count,
        indexed=indexed,
    )

    capsule: dict[str, str | int] = {
        "schema": SCHEMA,
        "mesa_driver": MESA_DRIVER,
        "manifest_index": int(row["index"]),
        "case": case_name,
        "width": width,
        "height": height,
        "rdc_sha256": rdc_sha256,
        "api_trace_sha256": _sha256(api_trace),
        "gallium_trace_sha256": _sha256(gallium_trace),
        "vertex_shader_sha256": hashlib.sha256(
            vertex_shader.encode("utf-8")
        ).hexdigest(),
        "fragment_shader_sha256": hashlib.sha256(
            fragment_shader.encode("utf-8")
        ).hexdigest(),
        "api_calls": api_calls,
        "api_draw_calls": 1,
        "primitive": primitive,
        "indexed": int(indexed),
        "draw_count": draw_count,
    }
    capsule.update(gallium)
    return capsule


CAPSULE_FIELD_ORDER = (
    "schema",
    "mesa_driver",
    "manifest_index",
    "case",
    "width",
    "height",
    "rdc_sha256",
    "api_trace_sha256",
    "gallium_trace_sha256",
    "vertex_shader_sha256",
    "fragment_shader_sha256",
    "api_calls",
    "api_draw_calls",
    "primitive",
    "indexed",
    "draw_count",
    "index_size",
    "index_buffer_hex",
    "gallium_draw_calls",
    "gallium_target_draws",
    "gallium_framebuffer_matches",
    "gallium_resource_creates",
    "gallium_state_calls",
    "gallium_constant_bytes",
    "framebuffer_format",
    "clear_color_bits",
    "depth_enabled",
    "depth_write",
    "depth_func",
    "blend_enabled",
    "blend_rgb_func",
    "blend_rgb_src_factor",
    "blend_rgb_dst_factor",
    "blend_alpha_func",
    "blend_alpha_src_factor",
    "blend_alpha_dst_factor",
    "front_ccw",
    "cull_face",
    "vertex_elements",
    "vertex_format",
    "vertex_stride",
    "vertex_offset",
    "vertex_buffer_map",
    "vertex_buffer_hex",
    "vertex_buffer_1_hex",
    "vertex_constant_hex",
    "fragment_constant_hex",
    "texture_format",
    "texture_width",
    "texture_height",
    "texture_levels",
    "texture_mip_sizes",
    "texture_mip_hex",
    "sampler_min_img_filter",
    "sampler_min_mip_filter",
    "sampler_mag_img_filter",
)


def format_capsule(capsule: dict[str, str | int]) -> str:
    if set(capsule) != set(CAPSULE_FIELD_ORDER):
        raise CapsuleError("internal capsule field mismatch")
    return "".join(f"{field}={capsule[field]}\n" for field in CAPSULE_FIELD_ORDER)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build one strict RenderDoc/Mesa POC command capsule."
    )
    parser.add_argument("--rdc", required=True, type=Path)
    parser.add_argument("--api-trace", required=True, type=Path)
    parser.add_argument("--gallium-trace", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--case", required=True)
    parser.add_argument("--width", required=True, type=_positive_int)
    parser.add_argument("--height", required=True, type=_positive_int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    options = parser.parse_args(argv)
    if not re.fullmatch(r"[a-z0-9_]+", options.case):
        parser.error("--case contains unsafe characters")
    try:
        capsule = build_capsule(
            rdc=options.rdc,
            api_trace=options.api_trace,
            gallium_trace=options.gallium_trace,
            manifest=options.manifest,
            case_name=options.case,
            width=options.width,
            height=options.height,
        )
        output = format_capsule(capsule)
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(output, encoding="utf-8")
    except (CapsuleError, OSError, UnicodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
