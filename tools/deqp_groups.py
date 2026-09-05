"""Stable dEQP group catalog for the PvrGPU live batch runner.

Selectors in this module are discovery-only.  A caller must expand them into
exact case names and pass only values accepted by :func:`is_exact_case` to an
executing ``pvrgpu-deqp`` process.  This preserves the current SystemC
one-command-per-process contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Iterable


EXACT_CASE_RE = re.compile(
    r"^dEQP-(?:EGL|GLES2|GLES3|GLES31)\.[A-Za-z0-9_.-]+$"
)

GLES3_UNAVAILABLE = (
    "The runner contains the GLES3 package, but all current PvrGPU EGL configs "
    "advertise ES2 only (no EGL_OPENGL_ES3_BIT_KHR). Mesa keeps the driver at "
    "ES2 because ES3 format, query, and primitive-restart prerequisites are "
    "not implemented."
)
GLES31_UNAVAILABLE = (
    "The runner contains the GLES31 package, but the PvrGPU EGL config has no "
    "ES3 bit. ES3.1 additionally needs a real compute, SSBO, shader-image, and "
    "atomic implementation that the current driver does not provide."
)


def es3_enabled() -> bool:
    """Mirror the driver switch: ES 3.0/3.1 caps are on unless PVRGPU_DISABLE_ES3 is set.

    The PCO driver advertises OpenGL ES 3.1 by default (pvrgpu_screen.c); the
    blocked reasons above describe the ES2-only surface that
    ``PVRGPU_DISABLE_ES3=1`` restores, so they apply only in that mode.
    """
    value = os.environ.get("PVRGPU_DISABLE_ES3", "")
    return not (value and value != "0")


GLES32_UNAVAILABLE = (
    "Geometry and tessellation shaders are OpenGL ES 3.2 features. The PCO "
    "driver advertises ES 3.1 at most, so these groups stay blocked whatever "
    "the ES3 switch says."
)

_GLES3_BLOCKED = None if es3_enabled() else GLES3_UNAVAILABLE
_GLES31_BLOCKED = None if es3_enabled() else GLES31_UNAVAILABLE


@dataclass(frozen=True)
class GroupSpec:
    """One user-visible group and its canonical discovery selectors."""

    id: str
    label: str
    suite: str
    selectors: tuple[str, ...]
    availability_reason: str | None = None
    locked_case_count: int = 0

    @property
    def available(self) -> bool:
        return self.availability_reason is None


GROUP_SPECS: tuple[GroupSpec, ...] = (
    GroupSpec(
        "egl-create-context",
        "EGL · Create context",
        "dEQP-EGL",
        ("dEQP-EGL.functional.create_context.*",),
        locked_case_count=22,
    ),
    GroupSpec(
        "egl-image",
        "EGL · Image",
        "dEQP-EGL",
        ("dEQP-EGL.functional.image.*",),
        locked_case_count=226,
    ),
    GroupSpec(
        "egl-robustness",
        "EGL · Robustness",
        "dEQP-EGL",
        (
            "dEQP-EGL.functional.robustness.*",
            "dEQP-EGL.functional.get_proc_address.extension.gl_ext_robustness",
            "dEQP-EGL.functional.get_proc_address.extension.gl_khr_robustness",
        ),
        locked_case_count=60,
    ),
    GroupSpec(
        "gles3-color-clear",
        "OpenGL ES 3.0 · Color clear",
        "dEQP-GLES3",
        ("dEQP-GLES3.functional.color_clear.*",),
        _GLES3_BLOCKED,
        locked_case_count=19,
    ),
    GroupSpec(
        "gles3-fbo",
        "OpenGL ES 3.0 · FBO",
        "dEQP-GLES3",
        ("dEQP-GLES3.functional.fbo.*",),
        _GLES3_BLOCKED,
        locked_case_count=2077,
    ),
    GroupSpec(
        "gles3-fragment-ops",
        "OpenGL ES 3.0 · Fragment operations",
        "dEQP-GLES3",
        ("dEQP-GLES3.functional.fragment_ops.*",),
        _GLES3_BLOCKED,
        locked_case_count=3176,
    ),
    GroupSpec(
        "gles3-instancing",
        "OpenGL ES 3.0 · Instancing",
        "dEQP-GLES3",
        ("dEQP-GLES3.functional.instanced.*",),
        _GLES3_BLOCKED,
        locked_case_count=45,
    ),
    GroupSpec(
        "gles3-rasterization-primitives",
        "OpenGL ES 3.0 · Rasterization primitives",
        "dEQP-GLES3",
        ("dEQP-GLES3.functional.rasterization.primitives.*",),
        _GLES3_BLOCKED,
        locked_case_count=10,
    ),
    GroupSpec(
        "gles3-scissor",
        "OpenGL ES 3.0 · Scissor",
        "dEQP-GLES3",
        ("dEQP-GLES3.functional.fragment_ops.scissor.*",),
        _GLES3_BLOCKED,
        locked_case_count=26,
    ),
    GroupSpec(
        "gles3-shader-builtins",
        "OpenGL ES 3.0 · Shader built-in functions",
        "dEQP-GLES3",
        ("dEQP-GLES3.functional.shaders.builtin_functions.*",),
        _GLES3_BLOCKED,
        locked_case_count=1730,
    ),
    GroupSpec(
        "gles3-texture-compressed",
        "OpenGL ES 3.0 · Compressed textures",
        "dEQP-GLES3",
        ("dEQP-GLES3.functional.texture.compressed.*",),
        _GLES3_BLOCKED,
        locked_case_count=322,
    ),
    GroupSpec(
        "gles3-texture-filtering",
        "OpenGL ES 3.0 · Texture filtering",
        "dEQP-GLES3",
        ("dEQP-GLES3.functional.texture.filtering.*",),
        _GLES3_BLOCKED,
        locked_case_count=1124,
    ),
    GroupSpec(
        "gles3-transform-feedback",
        "OpenGL ES 3.0 · Transform feedback",
        "dEQP-GLES3",
        ("dEQP-GLES3.functional.transform_feedback.*",),
        _GLES3_BLOCKED,
        locked_case_count=1320,
    ),
    GroupSpec(
        "gles3-ubo",
        "OpenGL ES 3.0 · Uniform buffer objects",
        "dEQP-GLES3",
        ("dEQP-GLES3.functional.ubo.*",),
        _GLES3_BLOCKED,
        locked_case_count=2357,
    ),
    GroupSpec(
        "gles3-vertex-arrays",
        "OpenGL ES 3.0 · Vertex arrays",
        "dEQP-GLES3",
        ("dEQP-GLES3.functional.vertex_arrays.*",),
        _GLES3_BLOCKED,
        locked_case_count=1005,
    ),
    GroupSpec(
        "gles3-stress-draw",
        "OpenGL ES 3.0 · Stress draw",
        "dEQP-GLES3",
        ("dEQP-GLES3.stress.draw.*",),
        _GLES3_BLOCKED,
        locked_case_count=74,
    ),
    GroupSpec(
        "gles3-stress-memory",
        "OpenGL ES 3.0 · Stress memory",
        "dEQP-GLES3",
        ("dEQP-GLES3.stress.memory.*",),
        _GLES3_BLOCKED,
        locked_case_count=80,
    ),
    GroupSpec(
        "gles3-stress-shaders",
        "OpenGL ES 3.0 · Stress shaders",
        "dEQP-GLES3",
        (
            "dEQP-GLES3.stress.long_shaders.*",
            "dEQP-GLES3.stress.long_running_shaders.*",
        ),
        _GLES3_BLOCKED,
        locked_case_count=40,
    ),
    GroupSpec(
        "gles31-compute-basic",
        "OpenGL ES 3.1 · Basic compute",
        "dEQP-GLES31",
        ("dEQP-GLES31.functional.compute.basic.*",),
        _GLES31_BLOCKED,
        locked_case_count=41,
    ),
    GroupSpec(
        "gles31-stress-draw-indirect",
        "OpenGL ES 3.1 · Draw-indirect stress",
        "dEQP-GLES31",
        ("dEQP-GLES31.stress.draw_indirect.*",),
        _GLES31_BLOCKED,
        locked_case_count=23,
    ),
    GroupSpec(
        "gles31-ssbo",
        "OpenGL ES 3.1 · Shader storage buffer objects",
        "dEQP-GLES31",
        ("dEQP-GLES31.functional.ssbo.*",),
        _GLES31_BLOCKED,
        locked_case_count=2061,
    ),
    GroupSpec(
        "gles31-texture-multisample",
        "OpenGL ES 3.1 · Multisample textures",
        "dEQP-GLES31",
        ("dEQP-GLES31.functional.texture.multisample.*",),
        _GLES31_BLOCKED,
        locked_case_count=157,
    ),
    GroupSpec(
        "gles32-geometry-shading",
        "OpenGL ES 3.2 · Geometry shading",
        "dEQP-GLES31",
        ("dEQP-GLES31.functional.geometry_shading.*",),
        GLES32_UNAVAILABLE,
        locked_case_count=207,
    ),
    GroupSpec(
        "gles32-tessellation",
        "OpenGL ES 3.2 · Tessellation",
        "dEQP-GLES31",
        ("dEQP-GLES31.functional.tessellation.*",),
        GLES32_UNAVAILABLE,
        locked_case_count=406,
    ),
)

GROUPS_BY_ID: dict[str, GroupSpec] = {group.id: group for group in GROUP_SPECS}


def is_exact_case(case_name: str) -> bool:
    """Return whether *case_name* is safe for one executing runner process."""

    return bool(EXACT_CASE_RE.fullmatch(case_name))


def get_group(group: GroupSpec | str) -> GroupSpec:
    """Resolve a group spec or stable group id."""

    if isinstance(group, GroupSpec):
        return group
    try:
        return GROUPS_BY_ID[group]
    except KeyError as error:
        raise KeyError(f"unknown dEQP group id: {group}") from error


def exact_case_belongs_to_group(group: GroupSpec | str, case_name: str) -> bool:
    """Check exact-case membership without treating user input as a glob."""

    spec = get_group(group)
    if not is_exact_case(case_name) or not case_name.startswith(spec.suite + "."):
        return False
    for selector in spec.selectors:
        if selector.endswith("*"):
            if case_name.startswith(selector[:-1]):
                return True
        elif case_name == selector:
            return True
    return False


def filter_exact_cases(
    group: GroupSpec | str,
    case_names: Iterable[str],
) -> tuple[str, ...]:
    """Return unique, ordered exact cases belonging to *group*.

    Wildcards, comma lists, unrelated suites, malformed discovery output, and
    duplicate names are discarded.  The returned tuple is therefore safe to
    schedule one item at a time with ``--deqp-case=<exact-name>``.
    """

    spec = get_group(group)
    selected: list[str] = []
    seen: set[str] = set()
    for value in case_names:
        case_name = value.strip()
        if case_name in seen or not exact_case_belongs_to_group(spec, case_name):
            continue
        seen.add(case_name)
        selected.append(case_name)
    return tuple(selected)


def _validate_catalog() -> None:
    if len(GROUP_SPECS) != 24 or len(GROUPS_BY_ID) != len(GROUP_SPECS):
        raise RuntimeError("dEQP group catalog must contain 24 unique ids")
    for group in GROUP_SPECS:
        if (
            not group.id
            or not group.label
            or not group.selectors
            or group.locked_case_count < 0
        ):
            raise RuntimeError(f"incomplete dEQP group spec: {group!r}")
        for selector in group.selectors:
            wildcard_count = selector.count("*")
            valid_selector = (
                wildcard_count == 0 and is_exact_case(selector)
            ) or (
                wildcard_count == 1
                and selector.endswith("*")
                and selector.startswith(group.suite + ".")
                and is_exact_case(selector[:-1] + "case")
            )
            if not valid_selector or "," in selector:
                raise RuntimeError(
                    f"invalid discovery selector for {group.id}: {selector}"
                )


_validate_catalog()
