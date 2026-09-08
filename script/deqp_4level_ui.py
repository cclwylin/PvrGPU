#!/usr/bin/env python3
"""Desktop front end for the four-tier dEQP regression (docs/dEQP_4level.md).

This is the tiered counterpart of ``script/deqp_dynamic_ui.py``.  Both drive the
same ``script/run_deqp_dynamic.sh`` -- the shell script stays the single source
of truth for the dynamic wiring (stock deqp-<module> binary ->
DYLD_LIBRARY_PATH -> PCO driver Mesa prefix -> dlopen'd PvrGPU SystemC bridge).

What this window adds on top of the dynamic UI is the structure documented in
``docs/dEQP_4level.md``:

  * the 30-group directory, with each group's case count and its L1/L2 quota;
  * the four tiers L1..L4, each of which covers all 30 groups and differs only
    in how many cases it takes per group;
  * the document's equidistant sampler (NOT an integer stride -- see
    :func:`sample_evenly` for why);
  * round-robin sharding with one output directory per shard, because the
    driver *appends* to its counter files and two shards sharing a directory
    interleave their events;
  * the two ``multisample.default_framebuffer`` subgroups that need
    ``rgba8888d24s8ms4`` instead of the default ``ms0`` config;
  * merging the shard summaries and reporting the QPA reason behind each
    failure, i.e. sections 4.4 and 4.5 of the document.

The 30-group directory lives here rather than in ``tools/deqp_groups.py``
because that module is still the older 24-group catalog; the document lists the
widening as outstanding work.  Nothing in this file keys behaviour on a case
name: the tiers select *which* cases run, never what their result should be.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import re
import sys


def _ensure_ui_python() -> None:
    """Re-exec direct launches with the configured PySide6 interpreter."""
    try:
        import PySide6  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name != "PySide6":
            raise
    else:
        return

    work_root = Path(
        os.environ.get(
            "PVRGPU_WORK_ROOT",
            str(Path.home() / "Downloads" / "_Codex" / "Working" / "PvrGPU"),
        )
    ).expanduser()
    configured_python = os.environ.get("PVRGPU_UI_PYTHON", "").strip()
    configured_venv = Path(
        os.environ.get("PVRGPU_UI_VENV", str(work_root / "venv"))
    ).expanduser()
    candidates = [
        Path(configured_python).expanduser() if configured_python else None,
        configured_venv / "bin" / "python",
    ]
    current_python = Path(os.path.abspath(sys.executable))
    for candidate in candidates:
        if candidate is None or not candidate.is_file() or not os.access(candidate, os.X_OK):
            continue
        candidate = Path(os.path.abspath(candidate))
        if candidate == current_python:
            continue
        os.execv(
            str(candidate),
            [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]],
        )

    raise SystemExit(
        "PySide6 is not installed for the current python3 and no usable UI "
        "interpreter was found. Set PVRGPU_UI_PYTHON or install "
        "requirements-ui.txt into the project venv."
    )


if __name__ == "__main__":
    _ensure_ui_python()


from PySide6.QtCore import QProcess, QProcessEnvironment, QSettings, QTimer, QUrl, Qt
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QDesktopServices,
    QFont,
    QPalette,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# ------------------------------------------------------------------------------
# Project layout
# ------------------------------------------------------------------------------
REPO_ROOT = Path(
    os.environ.get("PVRGPU_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).resolve()
RUNNER_SCRIPT = REPO_ROOT / "script" / "run_deqp_dynamic.sh"
TIER_DOC = REPO_ROOT / "docs" / "dEQP_4level.md"

EVENT_PREFIX = "PVRGPU_DYN "
LOG_LINE_LIMIT = 40_000
DIAGNOSTICS_MAX_CASES = 12
DIAGNOSTICS_MAX_SUMMARY_ROWS = 200
DIAGNOSTICS_LOG_LINES = 200
ARTIFACT_ROW_LIMIT = 5_000
# The live table keeps only a window of rows: L3 plans 30k cases and L4 90k, so
# holding every row costs memory and redraw time for scrollback nobody reads.
# state.rows keeps them all, and the merged summary.tsv is the full record.
RESULT_ROW_LIMIT = 5_000
# Reading one QPA per failure is a file open each; a whole-tier run can fail in
# the thousands, so the reason tally stops well before it becomes the slow part
# of drawing the dashboard.
QPA_REASON_LIMIT = 400

PASS_STATUSES = {"pass"}
SKIP_STATUSES = {"notsupported", "waiver"}
WARNING_STATUSES = {"qualitywarning", "compatibilitywarning"}

EXACT_CASE_RE = re.compile(r"^dEQP-(?:EGL|GLES2|GLES3|GLES31)\.[A-Za-z0-9_.-]+$")

SUITE_TO_MODULE = {
    "dEQP-EGL": "egl",
    "dEQP-GLES2": "gles2",
    "dEQP-GLES3": "gles3",
    "dEQP-GLES31": "gles31",
}

DEFAULT_GL_CONFIG = "rgba8888d24s8ms0"
MS4_GL_CONFIG = "rgba8888d24s8ms4"

# Section 5 of the document.  Both default_framebuffer subgroups measure the
# *default* framebuffer, so ms0 leaves them with nothing to look at; every other
# case in those groups asks for its own sample count and is unaffected.
CONFIG_OVERRIDES: tuple[tuple[str, str, str], ...] = (
    (
        "dEQP-GLES31.functional.multisample.default_framebuffer.",
        MS4_GL_CONFIG,
        "第 22 組 default_framebuffer（8 條）需要 ms4",
    ),
    (
        "dEQP-GLES3.functional.multisample.default_framebuffer.",
        MS4_GL_CONFIG,
        "第 30 組 default_framebuffer（16 條）需要 ms4",
    ),
)

# The document's measured per-case average over 456 runs / 6,349 cases.  Used
# only to put an order of magnitude next to a plan; it is never a budget the
# run is held to.
AVERAGE_SECONDS_PER_CASE = 0.66


# ------------------------------------------------------------------------------
# The 30-group directory (docs/dEQP_4level.md section 3)
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Group4:
    """One row of the document's 30-group table.

    ``selectors`` are prefixes, always ending in ``.``, so that
    ``stress.draw.`` cannot swallow ``stress.draw_indirect.``.  ``exact_cases``
    exists for the handful of members that are single cases rather than a
    subtree.
    """

    number: int
    label: str
    suite: str
    selectors: tuple[str, ...]
    total: int
    l1: int
    l2: int
    exact_cases: tuple[str, ...] = ()
    seconds_per_case: float = AVERAGE_SECONDS_PER_CASE
    note: str = ""

    @property
    def module(self) -> str:
        return SUITE_TO_MODULE[self.suite]

    @property
    def key(self) -> str:
        return f"g{self.number:02d}"

    def matches(self, case_name: str) -> bool:
        if case_name in self.exact_cases:
            return True
        return any(case_name.startswith(prefix) for prefix in self.selectors)

    def quota(self, tier_id: str) -> int:
        """How many cases this group contributes to *tier_id*."""
        if tier_id == "L1":
            return self.l1
        if tier_id == "L2":
            return self.l2
        return self.total


def _selectors(suite: str, *tails: str) -> tuple[str, ...]:
    return tuple(f"{suite}.{tail}" for tail in tails)


_EGL = "dEQP-EGL"
_GLES3 = "dEQP-GLES3"
_GLES31 = "dEQP-GLES31"

GROUPS_4LEVEL: tuple[Group4, ...] = (
    Group4(1, "EGL Create context", _EGL,
           _selectors(_EGL, "functional.create_context."), 22, 22, 22),
    Group4(2, "EGL Image", _EGL,
           _selectors(_EGL, "functional.image."), 226, 100, 226),
    Group4(3, "EGL Robustness", _EGL,
           _selectors(_EGL, "functional.robustness."), 60, 60, 60,
           exact_cases=(
               "dEQP-EGL.functional.get_proc_address.extension.gl_ext_robustness",
               "dEQP-EGL.functional.get_proc_address.extension.gl_khr_robustness",
           ),
           note="functional.robustness. 本身只有 58 條；文件的 60 條含兩條 "
                "get_proc_address 擴充查詢，與 tools/deqp_groups.py 一致"),
    Group4(4, "Color clear", _GLES3,
           _selectors(_GLES3, "functional.color_clear."), 19, 19, 19),
    Group4(5, "FBO", _GLES3,
           _selectors(_GLES3, "functional.fbo."), 2077, 100, 400),
    Group4(6, "Fragment ops", _GLES3,
           _selectors(_GLES3, "functional.fragment_ops."), 3176, 100, 400,
           # 3176 cases in the document's measured 46 minutes single-process.
           seconds_per_case=0.87,
           note="單行程約 46 分鐘，分片後約 7 分鐘；"
                "depth_stencil 621 條每條 2.14 秒，是這一組最貴的子群"),
    Group4(7, "Instancing", _GLES3,
           _selectors(_GLES3, "functional.instanced."), 45, 45, 45),
    Group4(8, "Rasterization", _GLES3,
           _selectors(_GLES3, "functional.rasterization."), 108, 100, 108),
    Group4(9, "Texture functions + derivate", _GLES3,
           _selectors(_GLES3,
                      "functional.shaders.texture_functions.",
                      "functional.shaders.derivate."), 1221, 100, 400),
    Group4(10, "Shader built-in functions", _GLES3,
           _selectors(_GLES3, "functional.shaders.builtin_functions."),
           1730, 100, 400),
    Group4(11, "Compressed textures", _GLES3,
           _selectors(_GLES3, "functional.texture.compressed."), 322, 30, 120,
           seconds_per_case=2.78,
           note="每條 2.78 秒，全目錄最貴；L1/L2 的配額因此被蓋在 30／120"),
    Group4(12, "Texture filtering", _GLES3,
           _selectors(_GLES3, "functional.texture.filtering."), 1124, 100, 400),
    Group4(13, "Transform feedback", _GLES3,
           _selectors(_GLES3, "functional.transform_feedback."), 1320, 100, 400),
    Group4(14, "UBO", _GLES3,
           _selectors(_GLES3, "functional.ubo."), 2357, 100, 400),
    Group4(15, "Vertex arrays", _GLES3,
           _selectors(_GLES3, "functional.vertex_arrays."), 1005, 100, 400),
    Group4(16, "Stress draw", _GLES3,
           _selectors(_GLES3, "stress.draw."), 74, 74, 74),
    Group4(17, "Stress memory", _GLES3,
           _selectors(_GLES3, "stress.memory."), 80, 80, 80),
    Group4(18, "Stress shaders", _GLES3,
           _selectors(_GLES3,
                      "stress.long_shaders.",
                      "stress.long_running_shaders."), 40, 40, 40),
    Group4(19, "Compute", _GLES31,
           _selectors(_GLES31, "functional.compute."), 195, 100, 195,
           note="子群 basic 41、shared_var 136、indirect_dispatch 18；"
                "與第 21 組 SSBO 共用 shader 寫記憶體的路徑"),
    Group4(20, "Draw indirect", _GLES31,
           _selectors(_GLES31,
                      "functional.draw_indirect.",
                      "stress.draw_indirect."), 244, 100, 244,
           note="221 條 functional 裡 79 條 compute_interop 綁在第 19 組上"),
    Group4(21, "SSBO", _GLES31,
           _selectors(_GLES31, "functional.ssbo."), 2061, 100, 400,
           note="layout 2007 條（97%）考的是 std430 位址與 stride，不是繪圖"),
    Group4(22, "Multisample", _GLES31,
           _selectors(_GLES31,
                      "functional.texture.multisample.",
                      "functional.shaders.sample_variables.",
                      "functional.shaders.multisample_interpolation.",
                      "functional.sample_shading.",
                      "functional.multisample."), 595, 100, 400,
           note="先跑第 30 組（基本 MSAA 光柵化）再跑這組；"
                "default_framebuffer 子群需要 ms4"),
    Group4(23, "Geometry shading", _GLES31,
           _selectors(_GLES31, "functional.geometry_shading."), 207, 100, 207),
    Group4(24, "Tessellation", _GLES31,
           _selectors(_GLES31, "functional.tessellation."), 406, 100, 400),
    Group4(25, "Shader operator", _GLES3,
           _selectors(_GLES3, "functional.shaders.operator."), 6478, 100, 400,
           note="全目錄最大的一組，L2 只看得到 6%，要跑完只有 L3"),
    Group4(26, "Shader matrix", _GLES3,
           _selectors(_GLES3, "functional.shaders.matrix."), 2646, 100, 400),
    Group4(27, "Texture wrap", _GLES3,
           _selectors(_GLES3, "functional.texture.wrap."), 1440, 100, 400),
    Group4(28, "Image load/store", _GLES31,
           _selectors(_GLES31, "functional.image_load_store."), 747, 100, 400),
    Group4(29, "Sync + atomic counter", _GLES31,
           _selectors(_GLES31,
                      "functional.synchronization.",
                      "functional.atomic_counter."), 402, 100, 400),
    Group4(30, "Basic MSAA", _GLES3,
           _selectors(_GLES3, "functional.multisample."), 64, 64, 64,
           note="default_framebuffer 子群（16 條）需要 ms4"),
)

GROUPS_BY_NUMBER: dict[int, Group4] = {g.number: g for g in GROUPS_4LEVEL}


# ------------------------------------------------------------------------------
# The four tiers (docs/dEQP_4level.md section 1)
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Tier:
    id: str
    label: str
    cases: int
    cpu_time: str
    wall_8_shards: str
    budget: str
    purpose: str
    shards: int
    log_images: str
    rule: str
    note: str = ""

    @property
    def whole_modules(self) -> bool:
        """L4 enumerates the modules themselves rather than the 30 groups."""
        return self.id == "L4"


TIERS: tuple[Tier, ...] = (
    Tier("L1", "L1 · Very Fast Regression", 2534, "20 min", "2.7 min", "5 min",
         "每次 commit 前", 8, "disable",
         "每組 min(N, 100) 條，等距取樣；第 11 組 compressed 蓋在 30",
         "30 組全涵蓋，其中 8 組案例數不足 100 已 100% 覆蓋。"),
    Tier("L2", "L2 · Fast Regression", 7904, "54 min", "8.6 min", "15 min",
         "修完一個 subsystem", 8, "disable",
         "每組 min(N, 400) 條，等距取樣；第 11 組 compressed 蓋在 120",
         "再多 5 組 100% 覆蓋（共 13 組）。要完整驗 compute 或 draw indirect，"
         "跑 L2 就夠，不必等 L3。"),
    Tier("L3", "L3 · Detail Regression", 30491, "4.0 h", "40 min", "1 h",
         "milestone、過夜", 8, "disable",
         "30 組目錄全跑",
         "只有 L3 跑得完的是第 5、6、9、10、12、13、14、15、21、22、24、25、"
         "26、27、28、29 組。"),
    Tier("L4", "L4 · Full Regression", 90360, "12.5 h", "不分片", "25 h",
         "版本釋出、認證前", 1, "enable",
         "四個 module 列舉出的全部案例，單行程、開啟存圖",
         "刻意不分片：認證等級的 run 要順序固定、不跨片干擾。"
         "L4 另外要帶跑 ctest 與 tools/rdc_counter_report.py。"),
)

TIERS_BY_ID: dict[str, Tier] = {tier.id: tier for tier in TIERS}

# L4 enumerates whole modules.  The discovery cache normally holds three files;
# gles2 is listed here because the document counts four modules, and the UI
# offers to enumerate whatever is missing.
L4_MODULES: tuple[str, ...] = ("egl", "gles2", "gles3", "gles31")
DISCOVERY_MODULES: tuple[str, ...] = ("egl", "gles2", "gles3", "gles31")


def _validate_catalog() -> None:
    """Fail loudly if the table drifts away from the document.

    The three tier totals are the document's own numbers.  If a group's case
    count is edited without editing the tier row, this raises at import time
    rather than silently running a different regression than the one named.
    """
    if len(GROUPS_4LEVEL) != 30 or len(GROUPS_BY_NUMBER) != 30:
        raise RuntimeError("the four-tier catalog must hold 30 unique groups")
    for group in GROUPS_4LEVEL:
        if not group.selectors and not group.exact_cases:
            raise RuntimeError(f"group {group.number} has no selector")
        for selector in group.selectors:
            if not selector.startswith(group.suite + ".") or not selector.endswith("."):
                raise RuntimeError(
                    f"group {group.number} selector must be a dotted prefix "
                    f"inside {group.suite}: {selector}"
                )
        if not 0 < group.l1 <= group.l2 <= group.total:
            raise RuntimeError(
                f"group {group.number} quotas must satisfy 0 < L1 <= L2 <= N"
            )
    totals = {
        "L1": sum(g.l1 for g in GROUPS_4LEVEL),
        "L2": sum(g.l2 for g in GROUPS_4LEVEL),
        "L3": sum(g.total for g in GROUPS_4LEVEL),
    }
    for tier_id, counted in totals.items():
        declared = TIERS_BY_ID[tier_id].cases
        if counted != declared:
            raise RuntimeError(
                f"{tier_id} sums to {counted} cases but docs/dEQP_4level.md "
                f"declares {declared}"
            )


_validate_catalog()


# ------------------------------------------------------------------------------
# Sampling, sharding, planning
# ------------------------------------------------------------------------------
def sample_evenly(cases: list[str], quota: int) -> list[str]:
    """Take *quota* cases spread across the whole list.

    This is the document's sampler (section 4.2) and deliberately NOT an
    integer stride.  ``cases[::len(cases)//quota]`` degenerates to a stride of 1
    whenever the group holds fewer than twice the quota, which silently turns
    the sample back into "the first N cases in alphabetical order" -- exactly
    the sample the quota exists to avoid.  Group 19 sampled that way misses all
    18 ``indirect_dispatch`` cases; sampled this way it keeps them in
    proportion.
    """
    count = len(cases)
    if count == 0 or quota <= 0:
        return []
    take = min(quota, count)
    return [cases[index * count // take] for index in range(take)]


def round_robin_shards(cases: list[str], shards: int) -> list[list[str]]:
    """Split round-robin, the way ``awk '{ print >> ("shard_" NR % n) }'`` does.

    Round-robin rather than contiguous blocks because cost is not spread
    evenly through a caselist: a contiguous split hands one shard the whole
    expensive subgroup and leaves the rest idle.
    """
    if shards < 1:
        shards = 1
    buckets: list[list[str]] = [[] for _ in range(shards)]
    for index, case in enumerate(cases):
        buckets[index % shards].append(case)
    return [bucket for bucket in buckets if bucket]


def gl_config_for(case_name: str, default_config: str) -> tuple[str, str]:
    """Return ``(config, reason)`` for one case; reason is empty when default."""
    for prefix, config, reason in CONFIG_OVERRIDES:
        if case_name.startswith(prefix):
            return config, reason
    return default_config, ""


def module_for_case(case_name: str) -> str:
    """The runner module a case belongs to.

    Split on the first dot rather than matching prefixes: ``dEQP-GLES3`` is a
    prefix of ``dEQP-GLES31``, so a plain ``startswith`` sends every ES3.1 case
    to the ES3 binary.
    """
    return SUITE_TO_MODULE.get(case_name.split(".", 1)[0], "")


def group_for_case(case_name: str) -> Group4 | None:
    for group in GROUPS_4LEVEL:
        if case_name.startswith(group.suite + ".") and group.matches(case_name):
            return group
    return None


@dataclass
class GroupPlan:
    """What one group contributes to a planned run."""

    group: Group4
    discovered: int = 0
    quota: int = 0
    selected: int = 0

    @property
    def drift(self) -> int:
        """Discovered minus the count the document records for this group."""
        return self.discovered - self.group.total


@dataclass
class Bucket:
    """One caselist that a runner process can accept.

    Keyed by module because ``run_deqp_dynamic.sh`` refuses a caselist that
    crosses modules, and by gl-config because the two default_framebuffer
    subgroups need a different one.
    """

    module: str
    gl_config: str
    cases: list[str] = field(default_factory=list)
    reason: str = ""

    @property
    def tag(self) -> str:
        """Directory-safe name.  Only an overridden config earns a suffix, so
        the ordinary run keeps the plain module name."""
        if not self.reason:
            return self.module
        return f"{self.module}-{self.gl_config}"


@dataclass
class RunPlan:
    """Everything decided before a single process is started."""

    name: str
    tier: Tier
    groups: list[GroupPlan] = field(default_factory=list)
    buckets: list[Bucket] = field(default_factory=list)
    shards: int = 1
    log_images: str = "disable"
    default_config: str = DEFAULT_GL_CONFIG
    description: str = ""

    @property
    def total_cases(self) -> int:
        return sum(len(bucket.cases) for bucket in self.buckets)

    def estimated_seconds(self) -> float:
        seconds = 0.0
        for entry in self.groups:
            seconds += entry.selected * entry.group.seconds_per_case
        if not self.groups:
            seconds = self.total_cases * AVERAGE_SECONDS_PER_CASE
        return seconds


def bucket_case(
    buckets: dict[tuple[str, str], Bucket],
    case_name: str,
    module: str,
    default_config: str,
) -> None:
    """Append one case to the (module, gl-config) caselist it belongs in."""
    config, reason = gl_config_for(case_name, default_config)
    key = (module, config)
    bucket = buckets.get(key)
    if bucket is None:
        bucket = Bucket(module=module, gl_config=config, reason=reason)
        buckets[key] = bucket
    elif reason and not bucket.reason:
        bucket.reason = reason
    bucket.cases.append(case_name)


def plan_from_cases(
    *,
    name: str,
    tier: Tier,
    cases: list[str],
    shards: int,
    log_images: str,
    default_config: str,
    description: str = "",
) -> RunPlan:
    """Plan an explicit list of cases: one exact case, or a caselist file.

    The cases are taken as given -- no quota, no sampling -- but they still go
    through the same bucketing, so a caselist that happens to contain a
    default_framebuffer case still gets its ms4 config, and one that mixes
    modules still ends up in separate runner processes rather than being
    rejected by the runner.
    """
    plan = RunPlan(
        name=name,
        tier=tier,
        shards=max(1, shards),
        log_images=log_images,
        default_config=default_config,
    )
    buckets: dict[tuple[str, str], Bucket] = {}
    counts: dict[int, int] = {}
    seen: set[str] = set()
    for case_name in cases:
        if case_name in seen:
            continue
        module = module_for_case(case_name)
        if not module:
            continue
        seen.add(case_name)
        bucket_case(buckets, case_name, module, default_config)
        owner = group_for_case(case_name)
        if owner is not None:
            counts[owner.number] = counts.get(owner.number, 0) + 1
    plan.buckets = sorted(
        buckets.values(), key=lambda bucket: (bucket.module, bucket.gl_config)
    )
    for number, count in sorted(counts.items()):
        group = GROUPS_BY_NUMBER[number]
        plan.groups.append(
            GroupPlan(group=group, discovered=count, quota=count, selected=count)
        )
    plan.description = description or f"{len(seen)} cases"
    return plan


def build_plan(
    *,
    name: str,
    tier: Tier,
    groups: list[Group4],
    discovery: dict[str, list[str]],
    shards: int,
    log_images: str,
    default_config: str,
    whole_modules: bool = False,
) -> RunPlan:
    """Turn a tier plus a set of groups into per-module, per-config caselists."""
    plan = RunPlan(
        name=name,
        tier=tier,
        shards=max(1, shards),
        log_images=log_images,
        default_config=default_config,
    )
    buckets: dict[tuple[str, str], Bucket] = {}

    def add(case_name: str, module: str) -> None:
        bucket_case(buckets, case_name, module, default_config)

    seen: set[str] = set()

    if whole_modules:
        # L4: every case the four modules enumerate, in discovery order.
        counts: dict[int, int] = {}
        for module in L4_MODULES:
            for case_name in discovery.get(module, ()):
                if case_name in seen:
                    continue
                seen.add(case_name)
                add(case_name, module)
                owner = group_for_case(case_name)
                if owner is not None:
                    counts[owner.number] = counts.get(owner.number, 0) + 1
        for group in GROUPS_4LEVEL:
            selected = counts.get(group.number, 0)
            plan.groups.append(
                GroupPlan(
                    group=group,
                    discovered=selected,
                    quota=selected,
                    selected=selected,
                )
            )
        plan.buckets = sorted(
            buckets.values(), key=lambda bucket: (bucket.module, bucket.gl_config)
        )
        plan.description = (
            f"{tier.id} · {', '.join(L4_MODULES)} 全部案例 · "
            f"{plan.total_cases} cases"
        )
        return plan

    for group in groups:
        pool = [
            case_name
            for case_name in discovery.get(group.module, ())
            if group.matches(case_name)
        ]
        quota = group.quota(tier.id)
        chosen = [name for name in sample_evenly(pool, quota) if name not in seen]
        seen.update(chosen)
        plan.groups.append(
            GroupPlan(
                group=group,
                discovered=len(pool),
                quota=quota,
                selected=len(chosen),
            )
        )
        for case_name in chosen:
            add(case_name, group.module)

    plan.buckets = sorted(
        buckets.values(), key=lambda bucket: (bucket.module, bucket.gl_config)
    )
    plan.description = (
        f"{tier.id} · {len(groups)} 組 · {plan.total_cases} cases · "
        f"{plan.shards} shard"
    )
    return plan


# ------------------------------------------------------------------------------
# Small helpers shared with script/deqp_dynamic_ui.py
# ------------------------------------------------------------------------------
ENV_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def load_local_env(path: Path) -> dict[str, str]:
    """Parse config/local.env well enough to seed the path fields."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values

    def expand(text: str) -> str:
        return re.sub(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}",
            lambda match: values.get(
                match.group(1), os.environ.get(match.group(1), "")
            ),
            text,
        )

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = ENV_LINE_RE.match(line)
        if not match:
            continue
        name, value = match.group(1), match.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] == "'":
            value = value[1:-1]
        else:
            if len(value) >= 2 and value[0] == value[-1] == '"':
                value = value[1:-1]
            value = expand(value)
        values[name] = value
    return values


def status_bucket(status: str) -> str:
    lowered = status.strip().lower()
    if lowered in PASS_STATUSES:
        return "pass"
    if lowered in SKIP_STATUSES:
        return "skip"
    if lowered in WARNING_STATUSES:
        return "warn"
    return "fail"


def format_bytes(size: int) -> str:
    value = float(size)
    for suffix in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or suffix == "GiB":
            return f"{value:.0f} {suffix}" if suffix == "B" else f"{value:.1f} {suffix}"
        value /= 1024.0
    return f"{size} B"


def format_duration(milliseconds: int) -> str:
    seconds = milliseconds / 1000.0
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, seconds = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {seconds:02d}s"


def format_seconds(seconds: float) -> str:
    return format_duration(int(seconds * 1000))


def read_caselist(path: Path) -> list[str]:
    """Read a discovery or caselist file, dropping comments and blanks."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    names: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            names.append(line)
    return names


QPA_RESULT_RE = re.compile(r'<Result StatusCode="([^"]*)">([^<]*)')


def qpa_reason(qpa_path: Path, byte_cap: int = 262_144) -> tuple[str, str]:
    """The ``<Result>`` line dEQP wrote for one case.

    Only the tail is read: the result element closes the log, and a case that
    logged a large image comparison would otherwise be pulled into memory in
    full just to read its last line.
    """
    try:
        size = qpa_path.stat().st_size
        with qpa_path.open("r", encoding="utf-8", errors="replace") as handle:
            if size > byte_cap:
                handle.seek(size - byte_cap)
                handle.readline()
            text = handle.read()
    except OSError:
        return ("", "")
    matches = QPA_RESULT_RE.findall(text)
    if not matches:
        return ("", "")
    code, detail = matches[-1]
    return (code, " ".join(detail.split()))


@dataclass
class CaseRow:
    index: int
    case_name: str
    status: str = "Running"
    exit_code: int | None = None
    duration_ms: int = 0
    case_dir: Path | None = None
    qpa: Path | None = None
    log: Path | None = None
    job_tag: str = ""
    group_number: int = 0

    @property
    def bucket(self) -> str:
        return status_bucket(self.status)

    def artifact_summary(self) -> str:
        if self.case_dir is None or not self.case_dir.is_dir():
            return "-"
        marks: list[str] = []
        jsonl = self.case_dir / "systemc.jsonl"
        if jsonl.is_file() and jsonl.stat().st_size > 0:
            try:
                done = '"type":"done"' in jsonl.read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                done = False
            marks.append("systemc✓" if done else "systemc…")
        if any((self.case_dir / "systemc").glob("*.png")):
            marks.append("png")
        command = self.case_dir / "driver-command.txt"
        if command.is_file() and command.stat().st_size > 0:
            marks.append("cmd")
        if (self.case_dir / "link.txt").is_file():
            marks.append("link")
        return " · ".join(marks) if marks else "-"


@dataclass
class Job:
    """One runner process: one shard of one bucket, one output directory."""

    tag: str
    module: str
    gl_config: str
    caselist: Path
    output_dir: Path
    cases: list[str]
    process: QProcess | None = None
    exit_code: int | None = None
    started_at: datetime | None = None
    finished: bool = False
    buffer: str = ""
    summary_path: Path | None = None


@dataclass
class RunState:
    plan: RunPlan | None = None
    run_dir: Path | None = None
    summary_path: Path | None = None
    module: str = ""
    runner: str = ""
    archive_dir: str = ""
    mesa_prefix: str = ""
    systemc_lib: str = ""
    host_arch: str = ""
    total: int = 0
    started_at: datetime | None = None
    duration_ms: int = 0
    rows: list[CaseRow] = field(default_factory=list)
    rows_by_case: dict[str, CaseRow] = field(default_factory=dict)


# ------------------------------------------------------------------------------
# Appearance
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Theme:
    """One coherent color set, so the window works in light and dark mode."""

    window: str
    surface: str
    field: str
    subtle: str
    border: str
    text: str
    muted: str
    accent: str
    accent_text: str
    ok: str
    bad: str
    warn: str
    skip: str
    ok_bg: str
    bad_bg: str
    warn_bg: str
    log_bg: str
    log_text: str


LIGHT_THEME = Theme(
    window="#f4f6f9", surface="#ffffff", field="#ffffff", subtle="#eef1f6",
    border="#d9dee7", text="#1f2430", muted="#5b6472",
    accent="#2f6feb", accent_text="#ffffff",
    ok="#1b7f3b", bad="#b3261e", warn="#a06000", skip="#5b6472",
    ok_bg="#e8f5ec", bad_bg="#fdecea", warn_bg="#fdf4e3",
    log_bg="#11151c", log_text="#dfe6f1",
)

DARK_THEME = Theme(
    window="#1b1e24", surface="#242830", field="#1a1d24", subtle="#2b303a",
    border="#39404c", text="#e7ebf3", muted="#9aa4b5",
    accent="#4d8bff", accent_text="#0b0e13",
    ok="#5cc07f", bad="#ff7168", warn="#f0b357", skip="#9aa4b5",
    ok_bg="#1e3327", bad_bg="#3a2321", warn_bg="#38301d",
    log_bg="#11151c", log_text="#dfe6f1",
)


def detect_theme(application) -> Theme:
    """Follow the desktop appearance instead of forcing one palette."""
    try:
        window_color = application.palette().color(QPalette.ColorRole.Window)
        if window_color.lightness() < 128:
            return DARK_THEME
    except Exception:  # pragma: no cover - defensive
        pass
    return LIGHT_THEME


DEFAULT_APPEARANCE = "light"


def resolve_theme(preference: str, system_theme: Theme) -> Theme:
    if preference == "light":
        return LIGHT_THEME
    if preference == "dark":
        return DARK_THEME
    return system_theme


def apply_palette(application, theme: Theme) -> None:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(theme.window))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(theme.text))
    palette.setColor(QPalette.ColorRole.Base, QColor(theme.field))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(theme.subtle))
    palette.setColor(QPalette.ColorRole.Text, QColor(theme.text))
    palette.setColor(QPalette.ColorRole.Button, QColor(theme.surface))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(theme.text))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(theme.surface))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(theme.text))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(theme.muted))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(theme.accent))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(theme.accent_text))
    application.setPalette(palette)


MODE_TIER = "整層（30 組）"
MODE_GROUP = "單一組 · 選層"
MODE_CUSTOM = "自訂單一 case"
MODE_CASELIST = "Caselist 檔案"


class MainWindow(QMainWindow):
    """Pick a tier, watch the shards run, then read the per-group dashboard."""

    def __init__(
        self,
        theme: Theme | None = None,
        *,
        system_theme: Theme | None = None,
        appearance: str = DEFAULT_APPEARANCE,
    ) -> None:
        super().__init__()
        self.system_theme = system_theme or theme or LIGHT_THEME
        self.appearance = appearance
        self.theme = theme or LIGHT_THEME
        self.setWindowTitle("PvrGPU dEQP · 四層回歸")
        self.resize(1520, 940)
        self.setMinimumSize(1180, 720)

        self.settings = QSettings("PvrGPU", "deqp-4level-ui")
        self.local_env = load_local_env(REPO_ROOT / "config" / "local.env")

        # phase: idle | check | discover | run
        self.phase = "idle"
        self.aux_process: QProcess | None = None
        self.aux_buffer = ""
        self.discovery_queue: list[str] = []
        self.discovery_target: str = ""
        self.after_discovery = None
        self.discovery: dict[str, list[str]] = {}

        self.state = RunState()
        self.plan: RunPlan | None = None
        self.run_dir: Path | None = None
        self.jobs: list[Job] = []
        self.pending_jobs: list[Job] = []
        self.active_jobs: dict[QProcess, Job] = {}
        self.cancelling = False
        self.last_exit_code: int | None = None
        self.last_phase = ""

        # Set per run by the Run buttons rather than by a widget, so it cannot
        # be left on from a previous run.
        self.stop_on_fail = False
        self.skip_passed = False
        # Accumulates across runs (NOT reset per run): every case whose most
        # recently observed status was Pass.
        self.known_pass_cases: set[str] = set()
        self.log_lines = 0
        # Remembers the user's Log images choice while L4 forces "enable",
        # so switching back to another tier restores it instead of leaving
        # every later run writing images.
        self._forced_log_images: str | None = None

        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.setInterval(500)
        self.elapsed_timer.timeout.connect(self._tick_elapsed)

        self._build_ui()
        self._build_menu()
        self._restore_settings()
        self._load_discovery_cache()
        self._update_mode_visibility()
        self._refresh_plan_tab()
        self._apply_style()

    # --------------------------------------------------------------------------
    # Construction
    # --------------------------------------------------------------------------
    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setChildrenCollapsible(False)
        splitter.setSizes([460, 1060])
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Ready")

    def _compact_combo(self, combo: QComboBox) -> None:
        combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        combo.setMinimumContentsLength(12)
        combo.view().setMinimumWidth(380)

    def _caption(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("caption")
        return label

    def _field(self, caption: str, widget: QWidget, tooltip: str = "") -> QWidget:
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._caption(caption))
        layout.addWidget(widget)
        if tooltip:
            widget.setToolTip(tooltip)
        return holder

    def _side_by_side(self, left: QWidget, right: QWidget) -> QWidget:
        holder = QWidget()
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(left, 1)
        layout.addWidget(right, 1)
        return holder

    def _note(self, text: str = "") -> QLabel:
        label = QLabel(text)
        label.setObjectName("note")
        label.setWordWrap(True)
        label.setMinimumWidth(1)
        return label

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("外觀")
        group = QActionGroup(self)
        group.setExclusive(True)
        self.theme_actions: dict[str, QAction] = {}
        for key, text in (("light", "淺色"), ("dark", "深色"), ("system", "跟隨系統")):
            action = QAction(text, self)
            action.setCheckable(True)
            action.setChecked(key == self.appearance)
            action.triggered.connect(
                lambda _checked=False, chosen=key: self.set_appearance(chosen)
            )
            group.addAction(action)
            menu.addAction(action)
            self.theme_actions[key] = action

        doc_menu = self.menuBar().addMenu("說明")
        open_doc = QAction("開啟 docs/dEQP_4level.md", self)
        open_doc.triggered.connect(lambda: self._open_path(TIER_DOC))
        doc_menu.addAction(open_doc)

    def set_appearance(self, preference: str) -> None:
        self.appearance = preference
        self.settings.setValue("appearance/theme", preference)
        self.theme = resolve_theme(preference, self.system_theme)
        application = QApplication.instance()
        if application is not None:
            apply_palette(application, self.theme)
        self._apply_style()
        for row in self.state.rows:
            if row.exit_code is not None:
                self._update_result_row(row)
        action = self.theme_actions.get(preference)
        if action is not None and not action.isChecked():
            action.setChecked(True)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("sidePanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(14, 14, 10, 4)
        panel_layout.setSpacing(12)
        panel_layout.addWidget(self._build_selection_group())
        panel_layout.addWidget(self._build_paths_group())
        panel_layout.addWidget(self._build_parameters_group())
        panel_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setObjectName("sideScroll")
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        container.setObjectName("sidePanel")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(scroll, 1)

        actions = QWidget()
        actions.setObjectName("sideActions")
        action_layout = QVBoxLayout(actions)
        action_layout.setContentsMargins(14, 10, 10, 12)
        action_layout.setSpacing(8)

        # A tier is a sweep, not a debug loop: the default button runs the whole
        # plan and keeps going past failures.  Stopping at the first failure is
        # the deliberate second choice here, the opposite of the single-case UI.
        primary_row = QHBoxLayout()
        primary_row.setSpacing(8)
        self.run_button = QPushButton("Run 這一層")
        self.run_button.setObjectName("primary")
        self.run_button.setMinimumHeight(36)
        self.run_button.setToolTip(
            "跑完整份計畫，某個 case 失敗不會中斷後面的（依 --keep-going 勾選）"
        )
        self.run_button.clicked.connect(lambda: self.start_run())
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setMinimumHeight(36)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_run)
        self.quit_button = QPushButton("Quit")
        self.quit_button.setObjectName("quit")
        self.quit_button.setMinimumHeight(36)
        self.quit_button.setToolTip("關閉視窗；若還在跑會先問要不要中止")
        self.quit_button.clicked.connect(self.close)
        primary_row.addWidget(self.run_button, 2)
        primary_row.addWidget(self.cancel_button, 1)
        primary_row.addWidget(self.quit_button, 1)
        action_layout.addLayout(primary_row)

        self.run_stop_button = QPushButton("Run && Stop if fail")
        self.run_stop_button.setMinimumHeight(32)
        self.run_stop_button.setToolTip(
            "第一個失敗就停：該分片自己停下，其餘分片也會一起中止，\n"
            "還沒開始的分片不會啟動。除錯用，不要拿來做回歸紀錄。"
        )
        self.run_stop_button.clicked.connect(lambda: self.start_run(stop_on_fail=True))
        action_layout.addWidget(self.run_stop_button)

        self.run_skip_passed_button = QPushButton("Run Skip if passed")
        self.run_skip_passed_button.setMinimumHeight(32)
        self.run_skip_passed_button.setToolTip(
            "只跑目前記錄裡還沒 pass 過的 case（跳過已知 pass 的）。\n"
            "跟 Clear Pass 搭配：按過 Clear Pass 之後這裡就會整份重跑。"
        )
        self.run_skip_passed_button.clicked.connect(
            lambda: self.start_run(skip_passed=True)
        )
        action_layout.addWidget(self.run_skip_passed_button)

        secondary_row = QHBoxLayout()
        secondary_row.setSpacing(8)
        self.preview_button = QPushButton("預覽計畫")
        self.preview_button.setMinimumHeight(28)
        self.preview_button.setToolTip("不執行任何東西，只算出這一層會跑哪些 case")
        self.preview_button.clicked.connect(self.preview_plan)
        self.copy_commands_button = QPushButton("複製等效指令")
        self.copy_commands_button.setMinimumHeight(28)
        self.copy_commands_button.setToolTip(
            "把這份計畫寫成 dEQP_4level.md 第四節那種可以直接貼進終端機的指令"
        )
        self.copy_commands_button.clicked.connect(self.copy_commands)
        secondary_row.addWidget(self.preview_button)
        secondary_row.addWidget(self.copy_commands_button)
        action_layout.addLayout(secondary_row)

        self.clear_pass_button = QPushButton("Clear Pass")
        self.clear_pass_button.setMinimumHeight(28)
        self.clear_pass_button.setToolTip(
            "忘記目前記錄的已知 pass case：清空「Run Skip if passed」用的名單，\n"
            "並把目前結果表裡的 Pass 那幾行移除。"
        )
        self.clear_pass_button.clicked.connect(self.clear_known_pass)
        action_layout.addWidget(self.clear_pass_button)
        layout.addWidget(actions)

        container.setMinimumWidth(380)
        container.setMaximumWidth(600)
        return container

    def _build_selection_group(self) -> QGroupBox:
        box = QGroupBox("選擇要跑的層")
        layout = QVBoxLayout(box)
        layout.setSpacing(10)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems([MODE_TIER, MODE_GROUP, MODE_CUSTOM, MODE_CASELIST])
        self.mode_combo.currentTextChanged.connect(self._on_selection_changed)
        self._compact_combo(self.mode_combo)
        layout.addWidget(self._field("模式", self.mode_combo))

        self.tier_combo = QComboBox()
        for tier in TIERS:
            self.tier_combo.addItem(
                f"{tier.label} · {tier.cases} cases · 預算 {tier.budget}", tier.id
            )
        self.tier_combo.currentIndexChanged.connect(self._on_tier_changed)
        self._compact_combo(self.tier_combo)
        self.tier_field = self._field("層", self.tier_combo)
        layout.addWidget(self.tier_field)

        self.tier_note = self._note()
        layout.addWidget(self.tier_note)

        self.group_combo = QComboBox()
        for group in GROUPS_4LEVEL:
            self.group_combo.addItem(
                f"{group.number:2d} · {group.label} · {group.total}", group.number
            )
        self.group_combo.currentIndexChanged.connect(self._on_selection_changed)
        self._compact_combo(self.group_combo)
        self.group_field = self._field("組（30 組目錄）", self.group_combo)
        layout.addWidget(self.group_field)

        self.group_note = self._note()
        layout.addWidget(self.group_note)

        self.custom_case_edit = QLineEdit(
            "dEQP-GLES31.functional.compute.basic.ubo_to_ssbo_single_group"
        )
        self.custom_case_edit.setMinimumWidth(150)
        self.custom_field = self._field(
            "Case（一個 exact case）",
            self.custom_case_edit,
            "第 4.3 節的單一案例除錯；建議同時把 Log images 設成 enable",
        )
        layout.addWidget(self.custom_field)

        caselist_row = QWidget()
        caselist_layout = QHBoxLayout(caselist_row)
        caselist_layout.setContentsMargins(0, 0, 0, 0)
        caselist_layout.setSpacing(6)
        self.caselist_edit = QLineEdit()
        self.caselist_edit.setMinimumWidth(120)
        caselist_button = QPushButton("…")
        caselist_button.setFixedWidth(34)
        caselist_button.clicked.connect(self._browse_caselist)
        caselist_layout.addWidget(self.caselist_edit, 1)
        caselist_layout.addWidget(caselist_button)
        self.caselist_field = self._field("Caselist 檔案", caselist_row)
        layout.addWidget(self.caselist_field)

        self.plan_note = self._note()
        self.plan_note.setObjectName("planNote")
        layout.addWidget(self.plan_note)
        return box

    def _path_field(
        self, caption: str, value: str, *, directory: bool, tooltip: str = ""
    ) -> tuple[QWidget, QLineEdit]:
        edit = QLineEdit(value)
        edit.setMinimumWidth(120)
        edit.setCursorPosition(0)
        if tooltip:
            edit.setToolTip(tooltip)
        button = QPushButton("…")
        button.setFixedWidth(34)

        def browse() -> None:
            start = edit.text().strip() or str(Path.home())
            if directory:
                chosen = QFileDialog.getExistingDirectory(self, caption, start)
            else:
                chosen, _ = QFileDialog.getOpenFileName(self, caption, start)
            if chosen:
                edit.setText(chosen)
                edit.setCursorPosition(0)

        button.clicked.connect(browse)
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        row_layout.addWidget(edit, 1)
        row_layout.addWidget(button)
        return self._field(caption, row, tooltip), edit

    def _build_paths_group(self) -> QGroupBox:
        box = QGroupBox("路徑與 discovery")
        layout = QVBoxLayout(box)
        layout.setSpacing(10)

        layout.addWidget(self._note("留空則沿用 config/local.env"))

        field, self.mesa_edit = self._path_field(
            "PCO driver（Mesa prefix）",
            self.local_env.get("PVRGPU_MESA_PVRGPU_PREFIX", ""),
            directory=True,
            tooltip="含 gallium pvrgpu 驅動與 libEGL / libGLESv2 的 Mesa prefix",
        )
        layout.addWidget(field)

        field, self.bridge_edit = self._path_field(
            "SystemC bridge（.dylib）",
            self.local_env.get("PVRGPU_SYSTEMC_API_LIB", ""),
            directory=False,
            tooltip="libpvrgpu_systemc_bridge.dylib，由驅動在執行時 dlopen",
        )
        layout.addWidget(field)

        field, self.deqp_binary_edit = self._path_field(
            "dEQP binary（可留空）",
            "",
            directory=False,
            tooltip="留空則自動由 dEQP 專案的 out/deqp-build-<arch>.env 找 deqp-<module>",
        )
        layout.addWidget(field)

        field, self.deqp_build_edit = self._path_field(
            "dEQP build dir",
            self.local_env.get("PVRGPU_DEQP_BUILD_DIR", ""),
            directory=True,
            tooltip="dEQP 的 CMake build 目錄（modules/<module>/deqp-<module>）",
        )
        layout.addWidget(field)

        field, self.output_edit = self._path_field(
            "Output root",
            self.local_env.get("PVRGPU_OUTPUT_ROOT", str(REPO_ROOT / "outputs")),
            directory=True,
            tooltip="discovery 快取放在 <output root>/deqp_groups/discovery",
        )
        self.output_edit.editingFinished.connect(self._load_discovery_cache)
        layout.addWidget(field)

        self.discovery_label = self._note()
        layout.addWidget(self.discovery_label)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        self.check_button = QPushButton("Preflight")
        self.check_button.setToolTip("只驗證接線，不執行任何 case")
        self.check_button.clicked.connect(self.start_check)
        self.discover_button = QPushButton("重新列舉")
        self.discover_button.setToolTip(
            "重新產生這次計畫需要的 discovery 檔案。\n"
            "dEQP 重建過就要重跑一次，否則 stride 取到的是另一批案例。"
        )
        self.discover_button.clicked.connect(self.rediscover)
        self.open_output_button = QPushButton("開啟輸出")
        self.open_output_button.clicked.connect(self._open_output_root)
        button_row.addWidget(self.check_button)
        button_row.addWidget(self.discover_button)
        button_row.addWidget(self.open_output_button)
        layout.addLayout(button_row)

        self.check_label = self._note("尚未驗證")
        layout.addWidget(self.check_label)
        return box

    def _build_parameters_group(self) -> QGroupBox:
        box = QGroupBox("執行參數")
        layout = QVBoxLayout(box)
        layout.setSpacing(10)

        self.gl_config_combo = QComboBox()
        self.gl_config_combo.setEditable(True)
        self.gl_config_combo.addItems(
            [DEFAULT_GL_CONFIG, MS4_GL_CONFIG, "rgb565d16s0ms0", ""]
        )
        layout.addWidget(
            self._field(
                "GL config（預設）",
                self.gl_config_combo,
                "第 22、30 組的 default_framebuffer 子群會自動改用 "
                f"{MS4_GL_CONFIG}，不吃這個欄位",
            )
        )
        layout.addWidget(
            self._note(
                f"例外：multisample.default_framebuffer 會自己切到 {MS4_GL_CONFIG}，"
                "並分到獨立的 caselist 與輸出目錄。"
            )
        )

        self.surface_combo = QComboBox()
        self.surface_combo.addItems(["pbuffer", "fbo", "window"])
        self.size_combo = QComboBox()
        self.size_combo.setEditable(True)
        self.size_combo.addItems(["256x256", "128x128", "64x64", "512x512"])
        layout.addWidget(
            self._side_by_side(
                self._field("Surface", self.surface_combo),
                self._field("Size", self.size_combo),
            )
        )

        self.log_images_combo = QComboBox()
        self.log_images_combo.addItems(["disable", "enable"])
        self.shards_spin = QSpinBox()
        self.shards_spin.setRange(1, 32)
        self.shards_spin.setValue(8)
        self.shards_spin.setToolTip(
            "L1 到 L3 都要靠分片才進得了預算；實測 8 片加速 6.2 倍。\n"
            "每片有自己的 --output-dir，因為驅動是附加寫入 counter 檔的，\n"
            "兩片寫進同一個目錄會讀到交錯的舊事件。"
        )
        # valueChanged hands the slot the new int; the plan tab takes no
        # argument, so drop it rather than letting it arrive as a plan.
        self.shards_spin.valueChanged.connect(lambda _value: self._refresh_plan_tab())
        layout.addWidget(
            self._side_by_side(
                self._field("Log images", self.log_images_combo),
                self._field("分片數", self.shards_spin),
            )
        )

        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(0, 86_400)
        self.timeout_spin.setSuffix(" s")
        self.timeout_spin.setSpecialValueText("無 timeout")
        layout.addWidget(self._field("Per-case timeout", self.timeout_spin))

        self.verify_link_check = QCheckBox("--verify-link")
        self.verify_link_check.setToolTip(
            "用 DYLD_PRINT_LIBRARIES 確認真的走到你 build 的 PCO driver"
        )
        self.keep_going_check = QCheckBox("--keep-going")
        self.keep_going_check.setChecked(True)
        layout.addWidget(self.verify_link_check)
        layout.addWidget(self.keep_going_check)
        layout.addWidget(
            self._note(
                "回歸掃描應該保持 --keep-going 勾選；只有「Run && Stop if fail」"
                "會刻意拿掉它。"
            )
        )
        return box

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 12, 12, 12)
        layout.setSpacing(8)

        header = QFrame()
        header.setObjectName("header")
        header_layout = QGridLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)

        self.current_case_label = QLabel("尚未執行")
        self.current_case_label.setObjectName("currentCase")
        header_layout.addWidget(self.current_case_label, 0, 0, 1, 4)

        self.copy_first_fail_button = QPushButton("複製第一個 Fail")
        self.copy_first_fail_button.setToolTip(
            "只把第一個 Fail 的 case（設定、命令與 log 尾巴）收成一份純文字，\n"
            "複製到剪貼簿，同時寫成 run 目錄裡的 diagnostics-first-fail.txt。\n"
            "沒有 Fail 的 case 時會提示，不會複製空白內容。"
        )
        self.copy_first_fail_button.clicked.connect(
            lambda: self.copy_diagnostics(only_first_fail=True)
        )
        header_layout.addWidget(self.copy_first_fail_button, 0, 4)

        self.copy_diagnostics_button = QPushButton("複製診斷資訊（全部）")
        self.copy_diagnostics_button.setToolTip(
            "把這次執行的計畫、解析後的路徑，以及有問題（fail/warn/\n"
            "NotSupported）case 的結果、命令與 log 尾巴收成一份純文字，\n"
            "複製到剪貼簿，同時寫成 run 目錄裡的 diagnostics.txt。"
        )
        self.copy_diagnostics_button.clicked.connect(
            lambda: self.copy_diagnostics(only_first_fail=False)
        )
        header_layout.addWidget(self.copy_diagnostics_button, 0, 5)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        header_layout.addWidget(self.progress, 1, 0, 1, 6)

        self.counter_labels: dict[str, QLabel] = {}
        for column, (key, text) in enumerate(
            (
                ("total", "Total"),
                ("pass", "Pass"),
                ("fail", "Fail"),
                ("skip", "Skip"),
                ("warn", "Warn"),
                ("elapsed", "Elapsed"),
            )
        ):
            tile = QLabel(f"{text}\n0")
            tile.setObjectName(f"tile-{key}")
            tile.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tile.setProperty("tile", True)
            tile.setMinimumHeight(52)
            self.counter_labels[key] = tile
            header_layout.addWidget(tile, 2, column)
        layout.addWidget(header)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_plan_tab(), "計畫")
        self.tabs.addTab(self._build_status_tab(), "Run status")
        self.tabs.addTab(self._build_groups_tab(), "分組結果")
        self.tabs.addTab(self._build_log_tab(), "Log")
        self.tabs.addTab(self._build_dashboard_tab(), "Dashboard")
        self.tabs.addTab(self._build_artifacts_tab(), "Artifacts")
        layout.addWidget(self.tabs, 1)
        return panel

    def _build_plan_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 10, 4, 4)
        layout.setSpacing(8)

        self.plan_title = QLabel("30 組目錄")
        self.plan_title.setObjectName("dashboardTitle")
        layout.addWidget(self.plan_title)
        layout.addWidget(
            self._note(
                "來源：docs/dEQP_4level.md 第三節。「discovery」是這台機器目前列舉到的"
                "條數，跟 N 不一樣就表示 CTS 換過，取樣會落在另一批案例上。"
            )
        )

        self.plan_table = QTableWidget(0, 8)
        self.plan_table.setHorizontalHeaderLabels(
            ["#", "組名", "Suite", "N", "L1", "L2", "discovery", "本次選取"]
        )
        self.plan_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.plan_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.plan_table.verticalHeader().setVisible(False)
        self.plan_table.setAlternatingRowColors(True)
        self.plan_table.verticalHeader().setDefaultSectionSize(24)
        header = self.plan_table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for column in (0, 2, 3, 4, 5, 6, 7):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.plan_table, 1)

        self.plan_detail = self._note()
        layout.addWidget(self.plan_detail)
        return widget

    def _build_status_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 8, 0, 0)
        self.results_table = QTableWidget(0, 7)
        self.results_table.setHorizontalHeaderLabels(
            ["#", "Case", "Status", "Exit", "Duration", "Shard", "Artifacts"]
        )
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.verticalHeader().setDefaultSectionSize(26)
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for column in range(2, 7):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.itemDoubleClicked.connect(self._open_case_item)
        layout.addWidget(self.results_table)

        self.follow_check = QCheckBox("自動捲到最新一列")
        self.follow_check.setChecked(True)
        layout.addWidget(self.follow_check)
        layout.addWidget(
            self._note(
                f"表格只留最近 {RESULT_ROW_LIMIT} 列；每一條的結果都在 run 目錄的"
                " summary.tsv 裡，統計與分組結果也不受影響。"
            )
        )
        return widget

    def _build_groups_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 8, 0, 0)
        self.groups_table = QTableWidget(0, 8)
        self.groups_table.setHorizontalHeaderLabels(
            ["#", "組名", "選取", "Pass", "Fail", "Skip", "Warn", "通過率"]
        )
        self.groups_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.groups_table.verticalHeader().setVisible(False)
        self.groups_table.setAlternatingRowColors(True)
        self.groups_table.verticalHeader().setDefaultSectionSize(24)
        header = self.groups_table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for column in (0, 2, 3, 4, 5, 6, 7):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.groups_table)
        layout.addWidget(
            self._note(
                "只統計這次跑到的案例。通過率是 Pass ÷ 已完成，"
                "NotSupported 不算通過也不算失敗。"
            )
        )
        return widget

    def _build_log_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 8, 0, 0)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(LOG_LINE_LIMIT)
        self.log_view.setFont(QFont("Menlo", 11))
        layout.addWidget(self.log_view)
        return widget

    def _build_dashboard_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 12, 4, 4)
        layout.setSpacing(12)

        self.dashboard_title = QLabel("跑完之後這裡會出現彙總")
        self.dashboard_title.setObjectName("dashboardTitle")
        layout.addWidget(self.dashboard_title)

        distribution = QGroupBox("結果分佈")
        distribution_layout = QGridLayout(distribution)
        self.distribution_bars: dict[str, tuple[QLabel, QProgressBar, QLabel]] = {}
        for row, (key, text) in enumerate(
            (("pass", "Pass"), ("fail", "Fail"), ("skip", "Skip"), ("warn", "Warning"))
        ):
            name = QLabel(text)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setObjectName(f"bar-{key}")
            count = QLabel("0")
            count.setMinimumWidth(90)
            distribution_layout.addWidget(name, row, 0)
            distribution_layout.addWidget(bar, row, 1)
            distribution_layout.addWidget(count, row, 2)
            self.distribution_bars[key] = (name, bar, count)
        layout.addWidget(distribution)

        reasons = QGroupBox("失敗原因（QPA 的 Result 欄）")
        reasons_layout = QVBoxLayout(reasons)
        self.reasons_table = QTableWidget(0, 3)
        self.reasons_table.setHorizontalHeaderLabels(["次數", "StatusCode", "訊息"])
        self.reasons_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        for column in (0, 1):
            self.reasons_table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        self.reasons_table.verticalHeader().setVisible(False)
        self.reasons_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.reasons_table.setMaximumHeight(220)
        reasons_layout.addWidget(self.reasons_table)
        layout.addWidget(reasons)

        shards = QGroupBox("分片")
        shards_layout = QVBoxLayout(shards)
        self.shards_table = QTableWidget(0, 5)
        self.shards_table.setHorizontalHeaderLabels(
            ["分片", "Module", "GL config", "Cases", "Exit"]
        )
        self.shards_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        for column in (1, 2, 3, 4):
            self.shards_table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        self.shards_table.verticalHeader().setVisible(False)
        self.shards_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.shards_table.setMaximumHeight(220)
        shards_layout.addWidget(self.shards_table)
        layout.addWidget(shards)

        wiring = QGroupBox("這次實際串到的東西")
        wiring_layout = QVBoxLayout(wiring)
        self.wiring_table = QTableWidget(0, 2)
        self.wiring_table.setHorizontalHeaderLabels(["項目", "值"])
        self.wiring_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.wiring_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.wiring_table.verticalHeader().setVisible(False)
        self.wiring_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.wiring_table.setMaximumHeight(230)
        wiring_layout.addWidget(self.wiring_table)
        layout.addWidget(wiring)

        slowest = QGroupBox("最慢的 case")
        slowest_layout = QVBoxLayout(slowest)
        self.slowest_table = QTableWidget(0, 3)
        self.slowest_table.setHorizontalHeaderLabels(["Case", "Status", "Duration"])
        self.slowest_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        for column in (1, 2):
            self.slowest_table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        self.slowest_table.verticalHeader().setVisible(False)
        self.slowest_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.slowest_table.setMaximumHeight(220)
        slowest_layout.addWidget(self.slowest_table)
        layout.addWidget(slowest)

        self.failure_label = self._note()
        layout.addWidget(self.failure_label)

        button_row = QHBoxLayout()
        self.open_summary_button = QPushButton("開啟合併後的 summary.tsv")
        self.open_summary_button.clicked.connect(self._open_summary)
        self.open_run_button = QPushButton("開啟這次的輸出目錄")
        self.open_run_button.clicked.connect(self._open_run_dir)
        button_row.addWidget(self.open_summary_button)
        button_row.addWidget(self.open_run_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        return scroll

    def _build_artifacts_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 8, 0, 0)
        self.artifacts_table = QTableWidget(0, 3)
        self.artifacts_table.setHorizontalHeaderLabels(["檔案", "類型", "大小"])
        self.artifacts_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.artifacts_table.verticalHeader().setVisible(False)
        self.artifacts_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.artifacts_table.itemDoubleClicked.connect(self._open_artifact_item)
        layout.addWidget(self.artifacts_table)
        layout.addWidget(
            self._note(
                f"最多列出 {ARTIFACT_ROW_LIMIT} 個檔案；完整內容在 run 目錄裡。"
            )
        )
        refresh = QPushButton("重新掃描")
        refresh.clicked.connect(self._refresh_artifacts)
        layout.addWidget(refresh)
        return widget

    # --------------------------------------------------------------------------
    # Selection
    # --------------------------------------------------------------------------
    def _current_tier(self) -> Tier:
        return TIERS_BY_ID.get(self.tier_combo.currentData(), TIERS[0])

    def _current_group(self) -> Group4:
        return GROUPS_BY_NUMBER.get(self.group_combo.currentData(), GROUPS_4LEVEL[0])

    def _selected_groups(self) -> list[Group4]:
        if self.mode_combo.currentText() == MODE_GROUP:
            return [self._current_group()]
        return list(GROUPS_4LEVEL)

    def _effective_shards(self) -> int:
        mode = self.mode_combo.currentText()
        if mode == MODE_CUSTOM:
            return 1
        if mode in (MODE_TIER, MODE_GROUP) and self._current_tier().id == "L4":
            # L4 is deliberately unsharded: a certification-grade run wants a
            # fixed order and no cross-shard interference.
            return 1
        return int(self.shards_spin.value())

    def _effective_log_images(self) -> str:
        if (
            self.mode_combo.currentText() in (MODE_TIER, MODE_GROUP)
            and self._current_tier().id == "L4"
        ):
            return "enable"
        return self.log_images_combo.currentText()

    def _update_mode_visibility(self) -> None:
        mode = self.mode_combo.currentText()
        self.tier_field.setVisible(mode in (MODE_TIER, MODE_GROUP))
        self.tier_note.setVisible(mode in (MODE_TIER, MODE_GROUP))
        self.group_field.setVisible(mode == MODE_GROUP)
        self.group_note.setVisible(mode == MODE_GROUP)
        self.custom_field.setVisible(mode == MODE_CUSTOM)
        self.caselist_field.setVisible(mode == MODE_CASELIST)

        forced = mode in (MODE_TIER, MODE_GROUP) and self._current_tier().id == "L4"
        self.shards_spin.setEnabled(not forced and mode != MODE_CUSTOM)
        self.log_images_combo.setEnabled(not forced)
        if forced:
            # Say it in the widget rather than only in a note, so the value the
            # run will use is the value on screen.
            if self._forced_log_images is None:
                self._forced_log_images = self.log_images_combo.currentText()
            self.log_images_combo.setCurrentText("enable")
        elif self._forced_log_images is not None:
            self.log_images_combo.setCurrentText(self._forced_log_images)
            self._forced_log_images = None

    def _on_tier_changed(self) -> None:
        self._update_mode_visibility()
        self._on_selection_changed()

    def _on_selection_changed(self) -> None:
        self._update_mode_visibility()
        self._update_tier_note()
        self._update_group_note()
        self._refresh_plan_tab()

    def _update_tier_note(self) -> None:
        tier = self._current_tier()
        text = (
            f"{tier.rule}。\n"
            f"用途：{tier.purpose} · 案例 {tier.cases} · CPU {tier.cpu_time} · "
            f"8 分片 {tier.wall_8_shards} · 預算 {tier.budget}。"
        )
        if tier.note:
            text += f"\n{tier.note}"
        if tier.id in ("L3", "L4"):
            # The one mistake that fakes a mass regression, from painful
            # experience: swapping the dylib underneath a running sweep.
            text += "\n跑這一層期間不要重建 driver 或 bridge：換掉執行中的 dylib 會" \
                    "造成 ABI 不一致，看起來像整批回歸。"
        self.tier_note.setText(text)

    def _update_group_note(self) -> None:
        group = self._current_group()
        tier = self._current_tier()
        discovered = len(self._group_pool(group))
        note = (
            f"{group.suite} · N={group.total} · L1={group.l1} · L2={group.l2}\n"
            f"selector：{', '.join(group.selectors) or '（只有列舉的單一 case）'}"
        )
        if group.exact_cases:
            note += f"\n另含 {len(group.exact_cases)} 條列舉的單一 case"
        note += f"\n這一層取 {min(group.quota(tier.id), max(discovered, 0))} 條"
        if discovered and discovered != group.total:
            note += f"\n⚠ discovery 目前列舉到 {discovered} 條，與文件的 {group.total} 不一致"
        if group.note:
            note += f"\n{group.note}"
        self.group_note.setText(note)

    def _sync_selection_to_settings(self) -> None:
        self.settings.setValue("selection/mode", self.mode_combo.currentText())
        self.settings.setValue("selection/tier", self.tier_combo.currentData())
        self.settings.setValue("selection/group", self.group_combo.currentData())
        self.settings.setValue("run/shards", self.shards_spin.value())

    def _browse_caselist(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Caselist", self.caselist_edit.text() or str(REPO_ROOT)
        )
        if chosen:
            self.caselist_edit.setText(chosen)
            self._refresh_plan_tab()

    # --------------------------------------------------------------------------
    # Discovery cache (docs/dEQP_4level.md section 2)
    # --------------------------------------------------------------------------
    def _output_root(self) -> Path:
        return Path(
            self.output_edit.text().strip()
            or self.local_env.get("PVRGPU_OUTPUT_ROOT", str(REPO_ROOT / "outputs"))
        ).expanduser()

    def _discovery_dir(self) -> Path:
        return self._output_root() / "deqp_groups" / "discovery"

    def _needed_modules(self) -> list[str]:
        mode = self.mode_combo.currentText()
        if mode == MODE_TIER:
            if self._current_tier().whole_modules:
                return list(L4_MODULES)
            return sorted({group.module for group in GROUPS_4LEVEL})
        if mode == MODE_GROUP:
            return [self._current_group().module]
        return []

    def _load_discovery_cache(self) -> None:
        """Re-read the three (or four) enumeration files from disk."""
        directory = self._discovery_dir()
        self.discovery = {}
        parts: list[str] = []
        for module in DISCOVERY_MODULES:
            path = directory / f"{module}.txt"
            if not path.is_file():
                continue
            cases = read_caselist(path)
            if not cases:
                continue
            self.discovery[module] = cases
            stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%m-%d %H:%M")
            parts.append(f"{module} {len(cases)}（{stamp}）")
        if parts:
            self.discovery_label.setText(
                f"discovery：{directory}\n" + " · ".join(parts)
            )
        else:
            self.discovery_label.setText(
                f"discovery：{directory}\n還沒有列舉檔案，按「重新列舉」產生。"
            )
        self._refresh_plan_tab()

    def _group_pool(self, group: Group4) -> list[str]:
        return [
            case_name
            for case_name in self.discovery.get(group.module, ())
            if group.matches(case_name)
        ]

    def _missing_modules(self) -> list[str]:
        return [
            module for module in self._needed_modules() if module not in self.discovery
        ]

    def rediscover(self) -> None:
        """Re-enumerate every module this plan needs, one process at a time."""
        modules = self._needed_modules() or list(DISCOVERY_MODULES)
        self._queue_discovery(modules, after=None, force=True)

    def _queue_discovery(self, modules, after, force: bool = False) -> None:
        if self.phase != "idle":
            QMessageBox.warning(self, "PvrGPU", "已經有一個執行中的工作。")
            return
        wanted = [
            module
            for module in modules
            if force or module not in self.discovery
        ]
        if not wanted:
            if after is not None:
                after()
            return
        self.discovery_queue = list(wanted)
        self.after_discovery = after
        self.tabs.setCurrentIndex(3)
        self._append_log(f"--- discovery queue: {', '.join(wanted)} ---")
        self._next_discovery()

    def _next_discovery(self) -> None:
        if not self.discovery_queue:
            after = self.after_discovery
            self.after_discovery = None
            self._load_discovery_cache()
            self._finish_idle()
            if after is not None:
                after()
            return
        module = self.discovery_queue.pop(0)
        self.discovery_target = module
        directory = self._discovery_dir()
        directory.mkdir(parents=True, exist_ok=True)
        arguments = self._common_arguments() + [
            "--module", module,
            "--discover",
            "--caselist-out", str(directory / f"{module}.txt"),
            "--output-dir", str(directory / f"_run_{module}"),
        ]
        self.current_case_label.setText(f"列舉 {module} …")
        self.statusBar().showMessage(f"Discovering {module}")
        self._start_aux(arguments, "discover")

    # --------------------------------------------------------------------------
    # Plan
    # --------------------------------------------------------------------------
    def _plan_name(self) -> str:
        mode = self.mode_combo.currentText()
        if mode == MODE_TIER:
            return self._current_tier().id
        if mode == MODE_GROUP:
            return f"g{self._current_group().number}_{self._current_tier().id}"
        if mode == MODE_CASELIST:
            stem = Path(self.caselist_edit.text().strip() or "caselist").stem
            return re.sub(r"[^A-Za-z0-9_.-]", "_", stem) or "caselist"
        return "single"

    def _make_plan(self, *, quiet: bool = False) -> RunPlan | None:
        """Turn the current selection into a concrete plan, or explain why not."""
        mode = self.mode_combo.currentText()
        tier = self._current_tier()
        shards = self._effective_shards()
        log_images = self._effective_log_images()
        default_config = self.gl_config_combo.currentText().strip() or DEFAULT_GL_CONFIG

        if mode in (MODE_TIER, MODE_GROUP):
            missing = self._missing_modules()
            if missing:
                if not quiet:
                    self._append_log(
                        f"--- discovery missing for: {', '.join(missing)} ---"
                    )
                return None
            return build_plan(
                name=self._plan_name(),
                tier=tier,
                groups=self._selected_groups(),
                discovery=self.discovery,
                shards=shards,
                log_images=log_images,
                default_config=default_config,
                whole_modules=mode == MODE_TIER and tier.whole_modules,
            )

        if mode == MODE_CUSTOM:
            case_name = self.custom_case_edit.text().strip()
            if not EXACT_CASE_RE.match(case_name):
                if not quiet:
                    QMessageBox.warning(
                        self, "PvrGPU", f"不是一個合法的 exact case：\n{case_name}"
                    )
                return None
            cases = [case_name]
        else:
            path = Path(self.caselist_edit.text().strip())
            if not path.is_file():
                if not quiet:
                    QMessageBox.warning(self, "PvrGPU", "請先選一個 caselist 檔案。")
                return None
            cases = [name for name in read_caselist(path) if EXACT_CASE_RE.match(name)]
            if not cases:
                if not quiet:
                    QMessageBox.warning(self, "PvrGPU", f"檔案裡沒有合法的 case：\n{path}")
                return None

        return plan_from_cases(
            name=self._plan_name(),
            tier=tier,
            cases=cases,
            shards=shards,
            log_images=log_images,
            default_config=default_config,
            description=f"{mode} · {len(cases)} cases",
        )

    def _refresh_plan_tab(self) -> None:
        """Redraw the 30-group table for the current selection."""
        if not hasattr(self, "plan_table"):
            return
        plan = self._make_plan(quiet=True)
        selected_by_number = {
            entry.group.number: entry.selected for entry in (plan.groups if plan else [])
        }
        mode = self.mode_combo.currentText()
        tier = self._current_tier()
        active = {group.number for group in self._selected_groups()}

        self.plan_table.setRowCount(len(GROUPS_4LEVEL))
        for row, group in enumerate(GROUPS_4LEVEL):
            discovered = len(self._group_pool(group))
            chosen = selected_by_number.get(group.number, 0)
            values = [
                str(group.number),
                group.label,
                group.suite.replace("dEQP-", ""),
                str(group.total),
                str(group.l1),
                str(group.l2),
                str(discovered) if discovered else "-",
                str(chosen) if chosen else "-",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 6 and discovered and discovered != group.total:
                    item.setForeground(QColor(self.theme.warn))
                if column == 7 and chosen:
                    item.setForeground(QColor(self.theme.ok))
                if group.number not in active and mode == MODE_GROUP:
                    item.setForeground(QColor(self.theme.muted))
                self.plan_table.setItem(row, column, item)

        title = f"30 組目錄 · {tier.label}" if mode in (MODE_TIER, MODE_GROUP) else "30 組目錄"
        self.plan_title.setText(title)

        details: list[str] = []
        if plan is None:
            missing = self._missing_modules()
            if missing:
                details.append(
                    f"缺少 discovery：{', '.join(missing)}。按「重新列舉」，"
                    "或直接按 Run，它會先幫你列舉再跑。"
                )
            else:
                details.append("目前的選擇還算不出計畫。")
        else:
            details.append(
                f"本次：{plan.total_cases} cases · {plan.shards} 分片 · "
                f"log-images {plan.log_images} · "
                f"估計 CPU {format_seconds(plan.estimated_seconds())}"
                f"（{plan.shards} 分片約 "
                f"{format_seconds(plan.estimated_seconds() / max(plan.shards, 1))}）"
            )
            for bucket in plan.buckets:
                line = f"  · {bucket.tag}：{len(bucket.cases)} cases · {bucket.gl_config}"
                if bucket.reason:
                    line += f"（{bucket.reason}）"
                details.append(line)
            drifted = [
                entry
                for entry in plan.groups
                if entry.selected and entry.discovered and entry.drift
            ]
            if drifted:
                details.append(
                    "⚠ 與文件記的 N 不一致的組："
                    + "、".join(
                        f"第 {entry.group.number} 組 {entry.discovered}"
                        f"（文件 {entry.group.total}）"
                        for entry in drifted[:6]
                    )
                )
        self.plan_detail.setText("\n".join(details))
        self._refresh_plan_note(plan)

    def _refresh_plan_note(self, plan: RunPlan | None = None) -> None:
        if not hasattr(self, "plan_note"):
            return
        if plan is None:
            plan = self._make_plan(quiet=True)
        if plan is None:
            missing = self._missing_modules()
            self.plan_note.setText(
                f"計畫尚未成立：缺少 {', '.join(missing)} 的 discovery。"
                if missing
                else "計畫尚未成立。"
            )
            return
        self.plan_note.setText(
            f"本次會跑 {plan.total_cases} cases，分成 {len(plan.buckets)} 份 caselist、"
            f"{plan.shards} 個分片；估計 "
            f"{format_seconds(plan.estimated_seconds() / max(plan.shards, 1))}。"
        )

    def preview_plan(self) -> None:
        missing = self._missing_modules()
        if missing:
            self._queue_discovery(missing, after=self._preview_after_discovery)
            return
        self._preview_after_discovery()

    def _preview_after_discovery(self) -> None:
        self._refresh_plan_tab()
        self.tabs.setCurrentIndex(0)
        plan = self._make_plan()
        if plan is not None:
            self.statusBar().showMessage(
                f"計畫：{plan.description} · {plan.total_cases} cases", 8000
            )

    # --------------------------------------------------------------------------
    # Command assembly
    # --------------------------------------------------------------------------
    def _common_arguments(self) -> list[str]:
        arguments: list[str] = ["--emit-events"]
        if self.mesa_edit.text().strip():
            arguments += ["--mesa-prefix", self.mesa_edit.text().strip()]
        if self.bridge_edit.text().strip():
            arguments += ["--systemc-lib", self.bridge_edit.text().strip()]
        if self.deqp_binary_edit.text().strip():
            arguments += ["--deqp-binary", self.deqp_binary_edit.text().strip()]
        elif self.deqp_build_edit.text().strip():
            arguments += ["--deqp-build-dir", self.deqp_build_edit.text().strip()]
        return arguments

    def _run_arguments(self, job: Job, plan: RunPlan) -> list[str]:
        arguments = self._common_arguments()
        arguments += ["--surface-type", self.surface_combo.currentText()]
        size = self.size_combo.currentText().strip()
        if re.fullmatch(r"[1-9][0-9]*x[1-9][0-9]*", size):
            arguments += ["--size", size]
        arguments += ["--gl-config", job.gl_config]
        arguments += ["--log-images", plan.log_images]
        if self.timeout_spin.value() > 0:
            arguments += ["--timeout", str(self.timeout_spin.value())]
        if self.verify_link_check.isChecked():
            arguments.append("--verify-link")
        # Stopping at the first failure is the absence of --keep-going, which
        # is what the runner already implements; the button just withholds it.
        if self.keep_going_check.isChecked() and not self.stop_on_fail:
            arguments.append("--keep-going")
        arguments += ["--caselist", str(job.caselist)]
        arguments += ["--output-dir", str(job.output_dir)]
        return arguments

    def build_commands(self, plan: RunPlan) -> str:
        """The same plan written as the shell commands of the document.

        The point is that this window stays a front end to the documented
        commands: whatever it just ran can be re-run by hand, one paste at a
        time, without this process.  The emitted script builds the same
        caselists the UI builds -- grep the group out of the discovery cache,
        sample it to the tier's quota, split off the ms4 subgroup -- so a paste
        of it runs the same cases, not an approximation of them.
        """
        lines: list[str] = []
        add = lines.append
        tmp = f"/tmp/{plan.name}"
        out = f"$PVRGPU_OUTPUT_ROOT/deqp_4level/{plan.name}"

        add("# 由 script/deqp_4level_ui.py 產生 —— 等效於 docs/dEQP_4level.md 第四節")
        add(f"# 計畫：{plan.description}")
        add("")
        add(f'cd "{REPO_ROOT}" \\')
        add("  && set -a && source config/local.env && set +a \\")
        add('  && export DISC="$PVRGPU_OUTPUT_ROOT/deqp_groups/discovery"')
        add("")

        modules = sorted({bucket.module for bucket in plan.buckets})

        if plan.tier.whole_modules:
            add("# L4：四個 module 列舉出的全部案例，直接用 discovery 快取")
            for module in modules:
                add(f'cp "$DISC/{module}.txt" {tmp}_{module}_all.txt')
        else:
            per_module: dict[str, list[str]] = {}
            for entry in plan.groups:
                if not entry.selected:
                    continue
                group = entry.group
                add(
                    f"# 第 {group.number} 組 {group.label}"
                    f"（N={group.total}，這一層取 {entry.quota}）"
                )
                add(
                    f"grep -E '{self._grep_pattern(group)}' \"$DISC/{group.module}.txt\""
                    f" > {tmp}_g{group.number}_all.txt"
                )
                if entry.quota < entry.discovered:
                    # Section 4.2's sampler, verbatim: equidistant, never an
                    # integer stride.
                    add(
                        "python3 -c \"import sys;c=[l.strip() for l in "
                        "open(sys.argv[1]) if l.strip()];"
                        "n=min(int(sys.argv[2]),len(c));"
                        "print('\\n'.join(c[i*len(c)//n] for i in range(n)))\""
                        f" {tmp}_g{group.number}_all.txt {entry.quota}"
                        f" > {tmp}_g{group.number}.txt"
                    )
                else:
                    add(
                        f"cp {tmp}_g{group.number}_all.txt "
                        f"{tmp}_g{group.number}.txt"
                    )
                per_module.setdefault(group.module, []).append(
                    f"{tmp}_g{group.number}.txt"
                )
            add("")
            for module in modules:
                files = per_module.get(module, [])
                if files:
                    add(f"cat {' '.join(files)} > {tmp}_{module}_all.txt")

        add("")
        add("# 第五節的 gl-config 例外：default_framebuffer 子群要走 ms4，")
        add("# 所以它自己一份 caselist、自己一個輸出目錄。")
        for module in modules:
            module_buckets = [b for b in plan.buckets if b.module == module]
            prefixes = [
                prefix
                for prefix, _config, _reason in CONFIG_OVERRIDES
                if module_for_case(prefix + "case") == module
            ]
            pattern = "|".join("^" + self._escape(prefix) for prefix in prefixes)
            for bucket in module_buckets:
                target = f"{tmp}_{bucket.tag}.txt"
                if len(module_buckets) == 1 or not pattern:
                    add(f"cp {tmp}_{module}_all.txt {target}")
                elif bucket.reason:
                    add(f"grep -E '{pattern}' {tmp}_{module}_all.txt > {target}")
                else:
                    add(f"grep -vE '{pattern}' {tmp}_{module}_all.txt > {target}")

        add("")
        add("# 每片一個 --output-dir：驅動是附加寫入 counter 檔的，")
        add("# 兩片寫進同一個目錄會讀到交錯的舊事件。")
        for bucket in plan.buckets:
            caselist = f"{tmp}_{bucket.tag}.txt"
            note = f"  # {bucket.reason}" if bucket.reason else ""
            add(f"# {bucket.tag}: {len(bucket.cases)} cases{note}")
            flags = self._doc_flags(plan, bucket)
            if plan.shards > 1:
                add(
                    f"rm -f {tmp}_{bucket.tag}_shard_* && "
                    f"awk -v n={plan.shards} '{{ f = \"{tmp}_"
                    f"{bucket.tag}_shard_\" (NR % n); print >> f }}' {caselist}"
                )
                add(
                    "for i in "
                    + " ".join(str(index) for index in range(plan.shards))
                    + "; do script/run_deqp_dynamic.sh"
                    f" --caselist {tmp}_{bucket.tag}_shard_$i {flags}"
                    f' --output-dir "{out}/{bucket.tag}/shard_$i" & done; wait'
                )
            else:
                add(
                    f"script/run_deqp_dynamic.sh --caselist {caselist} {flags}"
                    f' --output-dir "{out}/{bucket.tag}"'
                )
            add("")

        add("# 合併分片的 summary.tsv")
        add(
            f'cat "{out}"/*/summary.tsv "{out}"/*/*/summary.tsv 2>/dev/null'
            " | awk -F'\\t' 'NR==1 || $1!=\"case\"'"
            f' > "{out}/summary.tsv"'
        )
        add("")
        add("# 統計（第 4.5 節）")
        add(
            f"awk -F'\\t' 'NR>1 {{ c[$2]++ }} END {{ for (k in c) "
            f"printf \"%-18s %d\\n\", k, c[k] }}' \"{out}/summary.tsv\" | sort"
        )
        if plan.tier.id == "L4":
            add("")
            add("# L4 另外要帶跑：")
            add('ctest --test-dir "$PVRGPU_BUILD_DIR" --output-on-failure')
            add("python3 tools/rdc_counter_report.py")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _escape(text: str) -> str:
        """Escape for a grep -E pattern.

        Selectors hold only letters, digits, ``_``, ``-`` and ``.``, so the dot
        is the only metacharacter present -- and escaping just it keeps the
        emitted pattern identical to the ones written out in the document.
        """
        return text.replace(".", r"\.")

    def _grep_pattern(self, group: Group4) -> str:
        """One ERE matching a group: prefixes match a subtree, listed single
        cases are anchored so they cannot pick up a longer neighbour."""
        parts = ["^" + self._escape(selector) for selector in group.selectors]
        parts += ["^" + self._escape(case) + "$" for case in group.exact_cases]
        return "|".join(parts)

    def _doc_flags(self, plan: RunPlan, bucket: Bucket) -> str:
        parts = []
        if self.keep_going_check.isChecked() and not self.stop_on_fail:
            parts.append("--keep-going")
        parts += ["--surface-type", self.surface_combo.currentText()]
        size = self.size_combo.currentText().strip()
        if re.fullmatch(r"[1-9][0-9]*x[1-9][0-9]*", size):
            parts += ["--size", size]
        parts += ["--gl-config", bucket.gl_config]
        parts += ["--log-images", plan.log_images]
        if self.timeout_spin.value() > 0:
            parts += ["--timeout", str(self.timeout_spin.value())]
        return " ".join(parts)

    def copy_commands(self) -> None:
        plan = self._make_plan()
        if plan is None:
            QMessageBox.warning(
                self, "PvrGPU", "計畫還沒成立，先按「預覽計畫」或「重新列舉」。"
            )
            return
        text = self.build_commands(plan)
        QApplication.clipboard().setText(text)
        self.statusBar().showMessage(
            f"等效指令已複製（{format_bytes(len(text.encode('utf-8')))}）", 8000
        )

    # --------------------------------------------------------------------------
    # Preflight and discovery processes
    # --------------------------------------------------------------------------
    def _start_aux(self, arguments: list[str], phase: str) -> None:
        if self.aux_process is not None or self.active_jobs:
            QMessageBox.warning(self, "PvrGPU", "已經有一個執行中的工作。")
            return
        if not RUNNER_SCRIPT.is_file():
            QMessageBox.critical(self, "PvrGPU", f"找不到執行腳本：\n{RUNNER_SCRIPT}")
            return
        self.phase = phase
        self.aux_buffer = ""
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(environment)
        process.setWorkingDirectory(str(REPO_ROOT))
        process.readyReadStandardOutput.connect(self._read_aux_output)
        process.finished.connect(self._aux_finished)
        process.setProgram(str(RUNNER_SCRIPT))
        process.setArguments(arguments)
        self.aux_process = process
        self._append_log(f"$ {RUNNER_SCRIPT.name} {' '.join(arguments)}")
        self._set_busy(True)
        process.start()

    def _read_aux_output(self) -> None:
        if self.aux_process is None:
            return
        chunk = bytes(self.aux_process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        self.aux_buffer += chunk
        while "\n" in self.aux_buffer:
            line, self.aux_buffer = self.aux_buffer.split("\n", 1)
            self._handle_line(None, line.rstrip("\r"))

    def _aux_finished(self, exit_code: int, _status) -> None:
        if self.aux_buffer:
            self._handle_line(None, self.aux_buffer)
            self.aux_buffer = ""
        phase = self.phase
        self.last_phase = phase
        self.last_exit_code = exit_code
        self.aux_process = None
        self.phase = "idle"

        if phase == "check":
            if exit_code == 0:
                self.check_label.setText("✅ Preflight 通過：三個產物都在，架構一致。")
            else:
                self.check_label.setText("❌ Preflight 失敗，詳見 Log 分頁最後幾行。")
            self._finish_idle()
            return

        if phase == "discover":
            module = self.discovery_target
            if exit_code != 0:
                self.discovery_queue.clear()
                self.after_discovery = None
                self._load_discovery_cache()
                self._finish_idle()
                QMessageBox.critical(
                    self, "PvrGPU", f"列舉 {module} 失敗，詳見 Log 分頁。"
                )
                return
            self._append_log(f"--- discovery {module} done ---")
            self.phase = "discover"
            self._next_discovery()
            return

        self._finish_idle()

    def start_check(self) -> None:
        arguments = self._common_arguments() + ["--check", "--print-env"]
        modules = self._needed_modules()
        if modules:
            arguments += ["--module", modules[0]]
        else:
            plan = self._make_plan(quiet=True)
            if plan is not None and plan.buckets:
                arguments += ["--module", plan.buckets[0].module]
        self.check_label.setText("驗證中 …")
        self.tabs.setCurrentIndex(3)
        self._start_aux(arguments, "check")

    # --------------------------------------------------------------------------
    # Running
    # --------------------------------------------------------------------------
    def start_run(self, stop_on_fail: bool = False, skip_passed: bool = False) -> None:
        if self.aux_process is not None or self.active_jobs:
            QMessageBox.warning(self, "PvrGPU", "已經有一個執行中的工作。")
            return
        self.stop_on_fail = stop_on_fail
        self.skip_passed = skip_passed
        missing = self._missing_modules()
        if missing:
            self._queue_discovery(missing, after=self._start_run_now)
            return
        self._start_run_now()

    def _start_run_now(self) -> None:
        plan = self._make_plan()
        if plan is None:
            self._finish_idle()
            return

        if self.skip_passed and self.known_pass_cases:
            removed = 0
            for bucket in plan.buckets:
                before = len(bucket.cases)
                bucket.cases = [
                    name for name in bucket.cases if name not in self.known_pass_cases
                ]
                removed += before - len(bucket.cases)
            plan.buckets = [bucket for bucket in plan.buckets if bucket.cases]
            if removed:
                self._append_log(
                    f"--- skip-if-passed: {removed} case(s) already known passing ---"
                )
        if plan.total_cases == 0:
            QMessageBox.warning(
                self,
                "PvrGPU",
                "沒有可執行的 case（若剛按過 Run Skip if passed，代表都已標記為 pass）。",
            )
            self._finish_idle()
            return

        confirmation = self._confirm_long_run(plan)
        if not confirmation:
            self._finish_idle()
            return

        self.plan = plan
        self.run_dir = self._new_run_dir(plan)
        self._reset_run_view(plan)
        self._write_plan_files(plan)
        self._prepare_jobs(plan)
        if not self.jobs:
            QMessageBox.warning(self, "PvrGPU", "計畫沒有產生任何分片。")
            self._finish_idle()
            return

        self.phase = "run"
        self.cancelling = False
        self.state.started_at = datetime.now()
        self.elapsed_timer.start()
        self.tabs.setCurrentIndex(1)
        self.statusBar().showMessage(
            f"Running {plan.total_cases} case(s) in {len(self.jobs)} shard(s)"
        )
        self._set_busy(True)
        self._pump_jobs()

    def _confirm_long_run(self, plan: RunPlan) -> bool:
        """Ask once before a run measured in hours, and say why it is long."""
        estimate = plan.estimated_seconds() / max(plan.shards, 1)
        if estimate < 3600:
            return True
        answer = QMessageBox.question(
            self,
            "PvrGPU",
            f"{plan.description}\n\n"
            f"估計要跑 {format_seconds(estimate)}"
            f"（{plan.total_cases} cases，{plan.shards} 分片）。\n"
            f"文件給這一層的預算是 {plan.tier.budget}。\n\n"
            "跑的期間不要重建 driver 或 bridge。要開始嗎？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _new_run_dir(self, plan: RunPlan) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self._output_root() / "deqp_4level" / f"{stamp}_{plan.name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _write_plan_files(self, plan: RunPlan) -> None:
        assert self.run_dir is not None
        try:
            (self.run_dir / "commands.sh").write_text(
                self.build_commands(plan), encoding="utf-8"
            )
            payload = {
                "name": plan.name,
                "tier": plan.tier.id,
                "shards": plan.shards,
                "log_images": plan.log_images,
                "default_gl_config": plan.default_config,
                "total_cases": plan.total_cases,
                "buckets": [
                    {
                        "tag": bucket.tag,
                        "module": bucket.module,
                        "gl_config": bucket.gl_config,
                        "reason": bucket.reason,
                        "cases": len(bucket.cases),
                    }
                    for bucket in plan.buckets
                ],
                "groups": [
                    {
                        "number": entry.group.number,
                        "label": entry.group.label,
                        "documented": entry.group.total,
                        "discovered": entry.discovered,
                        "quota": entry.quota,
                        "selected": entry.selected,
                    }
                    for entry in plan.groups
                    if entry.selected
                ],
            }
            (self.run_dir / "plan.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as error:
            self._append_log(f"--- could not write plan files: {error} ---")

    def _prepare_jobs(self, plan: RunPlan) -> None:
        assert self.run_dir is not None
        caselist_dir = self.run_dir / "caselists"
        caselist_dir.mkdir(parents=True, exist_ok=True)
        self.jobs = []
        for bucket in plan.buckets:
            shards = round_robin_shards(bucket.cases, plan.shards)
            for index, shard_cases in enumerate(shards):
                tag = bucket.tag if len(shards) == 1 else f"{bucket.tag}-shard{index}"
                caselist = caselist_dir / f"{tag}.txt"
                caselist.write_text("\n".join(shard_cases) + "\n", encoding="utf-8")
                # One output directory per shard.  The driver appends to its
                # counter files, so two shards sharing a directory interleave
                # their events and the next reader believes a lie.
                output_dir = self.run_dir / "runs" / tag
                output_dir.mkdir(parents=True, exist_ok=True)
                self.jobs.append(
                    Job(
                        tag=tag,
                        module=bucket.module,
                        gl_config=bucket.gl_config,
                        caselist=caselist,
                        output_dir=output_dir,
                        cases=shard_cases,
                    )
                )
        self.pending_jobs = list(self.jobs)
        self._refresh_shards_table()

    def _pump_jobs(self) -> None:
        """Keep at most `shards` runner processes alive at a time."""
        if self.cancelling:
            return
        limit = max(1, self.plan.shards if self.plan else 1)
        while self.pending_jobs and len(self.active_jobs) < limit:
            self._start_job(self.pending_jobs.pop(0))

    def _start_job(self, job: Job) -> None:
        assert self.plan is not None
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(environment)
        process.setWorkingDirectory(str(REPO_ROOT))
        process.readyReadStandardOutput.connect(
            lambda bound=job: self._read_job_output(bound)
        )
        process.finished.connect(
            lambda code, _status, bound=job: self._job_finished(bound, code)
        )
        arguments = self._run_arguments(job, self.plan)
        process.setProgram(str(RUNNER_SCRIPT))
        process.setArguments(arguments)
        job.process = process
        job.started_at = datetime.now()
        self.active_jobs[process] = job
        self._append_log(f"$ [{job.tag}] {RUNNER_SCRIPT.name} {' '.join(arguments)}")
        process.start()

    def _read_job_output(self, job: Job) -> None:
        if job.process is None:
            return
        chunk = bytes(job.process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        job.buffer += chunk
        while "\n" in job.buffer:
            line, job.buffer = job.buffer.split("\n", 1)
            self._handle_line(job, line.rstrip("\r"))

    def _job_finished(self, job: Job, exit_code: int) -> None:
        if job.buffer:
            self._handle_line(job, job.buffer)
            job.buffer = ""
        job.exit_code = exit_code
        job.finished = True
        process = job.process
        job.process = None
        if process is not None:
            self.active_jobs.pop(process, None)
        self._append_log(f"--- [{job.tag}] finished, exit {exit_code} ---")
        self._reconcile_running_rows(job)
        self._refresh_shards_table()
        self._refresh_counters()
        self._pump_jobs()
        if not self.active_jobs and not self.pending_jobs:
            self._run_complete()

    def _reconcile_running_rows(self, job: Job) -> None:
        """A shard killed (stop-on-fail) or crashed mid-case never emits that
        case's case_end event, so its row is stuck at status="Running"
        forever -- which status_bucket() then falls through to "fail" for,
        even when the case actually passed (its own results.qpa says so).
        Re-derive the real outcome from disk before the row is used anywhere
        (counters, "first Fail", diagnostics)."""
        for row in self.state.rows:
            if row.job_tag != job.tag or row.status.strip().lower() != "running":
                continue
            qpa_path = row.qpa
            if qpa_path is None and row.case_dir is not None:
                qpa_path = row.case_dir / "results.qpa"
            code = ""
            if qpa_path is not None and Path(qpa_path).is_file():
                code, _ = qpa_reason(Path(qpa_path))
            if code:
                row.status = code
                row.qpa = Path(qpa_path)
                if row.bucket == "pass":
                    self.known_pass_cases.add(row.case_name)
                else:
                    self.known_pass_cases.discard(row.case_name)
            else:
                row.status = "Interrupted"
            row.exit_code = job.exit_code if job.exit_code is not None else -1
            self._update_result_row(row)

    def cancel_run(self) -> None:
        if self.aux_process is not None:
            self._append_log("--- cancel requested ---")
            self.aux_process.kill()
            self.discovery_queue.clear()
            self.after_discovery = None
            return
        if not self.active_jobs and not self.pending_jobs:
            return
        self.cancelling = True
        self.pending_jobs.clear()
        self._append_log("--- cancel requested, killing every shard ---")
        for process in list(self.active_jobs):
            process.kill()

    def _stop_after_failure(self, case_name: str) -> None:
        if self.cancelling:
            return
        self.cancelling = True
        pending = len(self.pending_jobs)
        self.pending_jobs.clear()
        self._append_log(
            f"--- stop-if-fail: {case_name} failed; killing "
            f"{len(self.active_jobs)} running shard(s), skipping {pending} queued ---"
        )
        for process in list(self.active_jobs):
            process.kill()

    def clear_known_pass(self) -> None:
        """Forget every case remembered as passing, and drop the Pass rows so
        the view agrees that nothing is recorded as passing."""
        self.known_pass_cases.clear()
        self.state.rows = [row for row in self.state.rows if row.bucket != "pass"]
        self.state.rows_by_case = {
            row.case_name: row for row in self.state.rows
        }
        for position in reversed(range(self.results_table.rowCount())):
            status_item = self.results_table.item(position, 2)
            if status_item is not None and status_bucket(status_item.text()) == "pass":
                self.results_table.removeRow(position)
        self._refresh_counters()
        self.statusBar().showMessage("已清除已知 Pass 記錄", 5000)

    # --------------------------------------------------------------------------
    # Process output
    # --------------------------------------------------------------------------
    def _handle_line(self, job: Job | None, line: str) -> None:
        if line.startswith(EVENT_PREFIX):
            try:
                payload = json.loads(line[len(EVENT_PREFIX) :])
            except json.JSONDecodeError:
                self._append_log(self._tag_line(job, line))
                return
            self._handle_event(job, payload)
            return
        self._append_log(self._tag_line(job, line))

    def _tag_line(self, job: Job | None, line: str) -> str:
        if job is None or len(self.jobs) <= 1:
            return line
        return f"[{job.tag}] {line}"

    def _handle_event(self, job: Job | None, payload: dict) -> None:
        event = payload.get("event", "")
        if event == "resolved":
            # Every shard resolves the same wiring; the first one to report it
            # is the one shown, and a disagreement would show up as a differing
            # exit code rather than a silently mixed run.
            if not self.state.runner:
                self.state.module = payload.get("module", "")
                self.state.runner = payload.get("runner", "")
                self.state.archive_dir = payload.get("archive_dir", "")
                self.state.mesa_prefix = payload.get("mesa_prefix", "")
                self.state.systemc_lib = payload.get("systemc_lib", "")
                self.state.host_arch = payload.get("host_arch", "")
                self._refresh_wiring_table()
        elif event == "check_ok":
            self.check_label.setText("✅ Preflight 通過：三個產物都在，架構一致。")
        elif event == "discovery_started":
            self._append_log(f"--- discovery: {payload.get('suite','')} ---")
        elif event == "discovery_finished":
            self._append_log(
                f"--- discovered {payload.get('discovered', 0)} case(s) ---"
            )
        elif event == "run_start":
            if job is not None:
                summary = payload.get("summary", "")
                job.summary_path = Path(summary) if summary else None
        elif event == "case_start":
            self._on_case_start(job, payload)
        elif event == "case_end":
            self._on_case_end(job, payload)
        elif event == "run_end":
            pass

    def _on_case_start(self, job: Job | None, payload: dict) -> None:
        case_name = payload.get("case", "")
        owner = group_for_case(case_name)
        row = CaseRow(
            index=len(self.state.rows) + 1,
            case_name=case_name,
            case_dir=Path(payload["case_dir"]) if payload.get("case_dir") else None,
            job_tag=job.tag if job else "",
            group_number=owner.number if owner else 0,
        )
        self.state.rows.append(row)
        self.state.rows_by_case[case_name] = row
        self._append_result_row(row)
        self._update_header(case_name)

    def _on_case_end(self, job: Job | None, payload: dict) -> None:
        case_name = payload.get("case", "")
        row = self.state.rows_by_case.get(case_name)
        if row is None:
            owner = group_for_case(case_name)
            row = CaseRow(
                index=len(self.state.rows) + 1,
                case_name=case_name,
                job_tag=job.tag if job else "",
                group_number=owner.number if owner else 0,
            )
            self.state.rows.append(row)
            self.state.rows_by_case[case_name] = row
            self._append_result_row(row)
        row.status = payload.get("status", "Unknown")
        row.exit_code = int(payload.get("exit_code", 0))
        row.duration_ms = int(payload.get("duration_ms", 0))
        for key in ("case_dir", "qpa", "log"):
            value = payload.get(key)
            if value:
                setattr(row, key, Path(value))
        # Track the most recently observed outcome, not "ever passed": a
        # regression must make a case eligible for "Skip if passed" again.
        if row.bucket == "pass":
            self.known_pass_cases.add(case_name)
        else:
            self.known_pass_cases.discard(case_name)
        self._update_result_row(row)
        self._refresh_counters()
        self._update_header()
        if self.stop_on_fail and row.bucket == "fail":
            self._stop_after_failure(case_name)

    def _update_header(self, current: str = "") -> None:
        finished = sum(1 for row in self.state.rows if row.exit_code is not None)
        total = self.state.total or len(self.state.rows)
        self.progress.setValue(min(finished, self.progress.maximum()))
        running = len(self.active_jobs)
        pieces = [f"[{finished}/{total}]"]
        if running:
            pieces.append(f"{running} 片執行中")
        if current:
            pieces.append(current)
        elif self.state.rows:
            pieces.append(self.state.rows[-1].case_name)
        self.current_case_label.setText(" · ".join(pieces))

    # --------------------------------------------------------------------------
    # Views
    # --------------------------------------------------------------------------
    def _reset_run_view(self, plan: RunPlan) -> None:
        self.state = RunState(
            plan=plan, run_dir=self.run_dir, total=plan.total_cases
        )
        self.results_table.setRowCount(0)
        self.groups_table.setRowCount(0)
        self.reasons_table.setRowCount(0)
        self.log_view.clear()
        self.log_lines = 0
        self.progress.setRange(0, max(1, plan.total_cases))
        self.progress.setValue(0)
        self.current_case_label.setText("準備中 …")
        self.failure_label.setText("")
        self._refresh_counters()

    def _append_log(self, text: str) -> None:
        if not text:
            return
        self.log_view.appendPlainText(text)
        self.log_lines += 1

    def _append_result_row(self, row: CaseRow) -> None:
        while self.results_table.rowCount() >= RESULT_ROW_LIMIT:
            # Drop from the top: the oldest rows are finished, and a case still
            # running is always near the bottom whatever the shard count.
            self.results_table.removeRow(0)
        index = self.results_table.rowCount()
        self.results_table.insertRow(index)
        for column, value in enumerate(
            (str(row.index), row.case_name, row.status, "", "", row.job_tag, "")
        ):
            item = QTableWidgetItem(value)
            if column == 1:
                item.setData(Qt.ItemDataRole.UserRole, str(row.case_dir or ""))
            self.results_table.setItem(index, column, item)
        if self.follow_check.isChecked():
            self.results_table.scrollToBottom()

    def _row_position(self, row: CaseRow) -> int | None:
        for position in range(self.results_table.rowCount() - 1, -1, -1):
            item = self.results_table.item(position, 1)
            if item is not None and item.text() == row.case_name:
                return position
        return None

    def _update_result_row(self, row: CaseRow) -> None:
        position = self._row_position(row)
        if position is None:
            return
        values = (
            str(row.index),
            row.case_name,
            row.status,
            "-" if row.exit_code is None else str(row.exit_code),
            f"{row.duration_ms} ms",
            row.job_tag,
            row.artifact_summary(),
        )
        for column, value in enumerate(values):
            item = self.results_table.item(position, column)
            if item is None:
                item = QTableWidgetItem()
                self.results_table.setItem(position, column, item)
            item.setText(value)
            if column == 1:
                item.setData(Qt.ItemDataRole.UserRole, str(row.case_dir or ""))
        status_item = self.results_table.item(position, 2)
        colors = {
            "pass": QColor(self.theme.ok),
            "fail": QColor(self.theme.bad),
            "skip": QColor(self.theme.skip),
            "warn": QColor(self.theme.warn),
        }
        status_item.setForeground(colors.get(row.bucket, QColor(self.theme.text)))

    def _refresh_counters(self) -> None:
        buckets = {"pass": 0, "fail": 0, "skip": 0, "warn": 0}
        for row in self.state.rows:
            if row.exit_code is None:
                continue
            buckets[row.bucket] += 1
        self.counter_labels["total"].setText(f"Total\n{self.state.total}")
        for key in buckets:
            self.counter_labels[key].setText(f"{key.capitalize()}\n{buckets[key]}")
        self._tick_elapsed()

    def _tick_elapsed(self) -> None:
        if self.state.started_at is None:
            self.counter_labels["elapsed"].setText("Elapsed\n-")
            return
        elapsed = (datetime.now() - self.state.started_at).total_seconds()
        self.counter_labels["elapsed"].setText(
            f"Elapsed\n{format_duration(int(elapsed * 1000))}"
        )

    def _refresh_shards_table(self) -> None:
        self.shards_table.setRowCount(len(self.jobs))
        for row, job in enumerate(self.jobs):
            state = "-"
            if job.finished:
                state = str(job.exit_code)
            elif job.process is not None:
                state = "跑"
            values = (
                job.tag,
                job.module,
                job.gl_config,
                str(len(job.cases)),
                state,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 4 and job.finished and job.exit_code:
                    item.setForeground(QColor(self.theme.bad))
                self.shards_table.setItem(row, column, item)

    def _refresh_groups_table(self) -> None:
        counts: dict[int, dict[str, int]] = {}
        for row in self.state.rows:
            if row.exit_code is None:
                continue
            entry = counts.setdefault(
                row.group_number, {"pass": 0, "fail": 0, "skip": 0, "warn": 0}
            )
            entry[row.bucket] += 1
        planned = {
            entry.group.number: entry.selected
            for entry in (self.state.plan.groups if self.state.plan else [])
        }
        numbers = sorted(set(counts) | {n for n, value in planned.items() if value})
        self.groups_table.setRowCount(len(numbers))
        for row, number in enumerate(numbers):
            group = GROUPS_BY_NUMBER.get(number)
            entry = counts.get(number, {"pass": 0, "fail": 0, "skip": 0, "warn": 0})
            finished = sum(entry.values())
            rate = f"{100.0 * entry['pass'] / finished:.1f}%" if finished else "-"
            values = (
                str(number) if number else "-",
                group.label if group else "（不在 30 組目錄內）",
                str(planned.get(number, finished)),
                str(entry["pass"]),
                str(entry["fail"]),
                str(entry["skip"]),
                str(entry["warn"]),
                rate,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 4 and entry["fail"]:
                    item.setForeground(QColor(self.theme.bad))
                if column == 7 and finished and entry["fail"] == 0:
                    item.setForeground(QColor(self.theme.ok))
                self.groups_table.setItem(row, column, item)

    def _refresh_wiring_table(self) -> None:
        plan = self.state.plan
        entries = [
            ("Plan", plan.description if plan else "-"),
            ("Tier", f"{plan.tier.label}（預算 {plan.tier.budget}）" if plan else "-"),
            ("Module", self.state.module),
            ("dEQP runner", self.state.runner),
            ("Archive dir", self.state.archive_dir),
            ("PCO driver (Mesa)", self.state.mesa_prefix),
            ("SystemC bridge", self.state.systemc_lib),
            ("Host arch", self.state.host_arch),
            ("Output", str(self.run_dir or "")),
        ]
        self.wiring_table.setRowCount(len(entries))
        for row, (name, value) in enumerate(entries):
            self.wiring_table.setItem(row, 0, QTableWidgetItem(name))
            self.wiring_table.setItem(row, 1, QTableWidgetItem(value))

    def _refresh_reasons_table(self) -> None:
        """Section 4.5: what the QPA says behind each non-pass result."""
        notable = [
            row
            for row in self.state.rows
            if row.exit_code is not None and row.bucket in ("fail", "warn")
        ]
        tally: dict[tuple[str, str], int] = {}
        for row in notable[:QPA_REASON_LIMIT]:
            qpa = row.qpa or (row.case_dir / "results.qpa" if row.case_dir else None)
            if qpa is None or not Path(qpa).is_file():
                key = ("", "沒有 results.qpa")
            else:
                code, detail = qpa_reason(Path(qpa))
                key = (code or row.status, detail or "（Result 沒有訊息）")
            tally[key] = tally.get(key, 0) + 1
        ordered = sorted(tally.items(), key=lambda item: item[1], reverse=True)
        self.reasons_table.setRowCount(len(ordered))
        for row, ((code, detail), count) in enumerate(ordered):
            self.reasons_table.setItem(row, 0, QTableWidgetItem(str(count)))
            self.reasons_table.setItem(row, 1, QTableWidgetItem(code))
            self.reasons_table.setItem(row, 2, QTableWidgetItem(detail[:400]))
        if len(notable) > QPA_REASON_LIMIT:
            self._append_log(
                f"--- QPA 原因只統計了前 {QPA_REASON_LIMIT} 個（共 {len(notable)} 個）---"
            )

    def _refresh_dashboard(self) -> None:
        finished = [row for row in self.state.rows if row.exit_code is not None]
        total = len(finished)
        buckets = {"pass": 0, "fail": 0, "skip": 0, "warn": 0}
        for row in finished:
            buckets[row.bucket] += 1

        stamp = (
            self.state.started_at.strftime("%Y-%m-%d %H:%M:%S")
            if self.state.started_at
            else "-"
        )
        plan = self.state.plan
        self.dashboard_title.setText(
            f"{stamp} · {plan.name if plan else '-'} · "
            f"{total}/{self.state.total} cases · "
            f"{format_duration(self.state.duration_ms)}"
        )

        for key, (_, bar, count) in self.distribution_bars.items():
            value = buckets[key]
            percent = int(round(100.0 * value / total)) if total else 0
            bar.setValue(percent)
            count.setText(f"{value}  ({percent}%)")

        slowest = sorted(finished, key=lambda item: item.duration_ms, reverse=True)[:10]
        self.slowest_table.setRowCount(len(slowest))
        for row, item in enumerate(slowest):
            self.slowest_table.setItem(row, 0, QTableWidgetItem(item.case_name))
            self.slowest_table.setItem(row, 1, QTableWidgetItem(item.status))
            self.slowest_table.setItem(
                row, 2, QTableWidgetItem(format_duration(item.duration_ms))
            )

        failures = [row for row in finished if row.bucket == "fail"]
        if failures:
            listed = "\n".join(
                f"  · {item.case_name} → {item.status} (exit {item.exit_code})"
                for item in failures[:12]
            )
            more = "" if len(failures) <= 12 else f"\n  … 另外 {len(failures) - 12} 個"
            self.failure_label.setText(f"未通過的 case：\n{listed}{more}")
        else:
            self.failure_label.setText("沒有未通過的 case。")

        self._refresh_wiring_table()
        self._refresh_groups_table()
        self._refresh_reasons_table()
        self._refresh_shards_table()
        self._refresh_artifacts()

    def _refresh_artifacts(self) -> None:
        root = self.run_dir
        self.artifacts_table.setRowCount(0)
        if root is None or not Path(root).is_dir():
            return
        root = Path(root)
        rows = 0
        # Walk lazily and stop at the cap.  sorted(rglob("*")) would first build
        # a list of every file in the run -- a tier run has one directory per
        # case, so that is hundreds of thousands of paths before the first row.
        for path in root.rglob("*"):
            if rows >= ARTIFACT_ROW_LIMIT:
                break
            if not path.is_file():
                continue
            self.artifacts_table.insertRow(rows)
            name_item = QTableWidgetItem(str(path.relative_to(root)))
            name_item.setData(Qt.ItemDataRole.UserRole, str(path))
            self.artifacts_table.setItem(rows, 0, name_item)
            self.artifacts_table.setItem(
                rows, 1, QTableWidgetItem(path.suffix.lstrip(".") or "file")
            )
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            self.artifacts_table.setItem(rows, 2, QTableWidgetItem(format_bytes(size)))
            rows += 1

    # --------------------------------------------------------------------------
    # Completion
    # --------------------------------------------------------------------------
    def _run_complete(self) -> None:
        self.phase = "idle"
        self.elapsed_timer.stop()
        if self.state.started_at is not None:
            self.state.duration_ms = int(
                (datetime.now() - self.state.started_at).total_seconds() * 1000
            )
        self._merge_summaries()
        self._write_stats()
        self._refresh_counters()
        self._refresh_dashboard()
        self.tabs.setCurrentIndex(4)

        finished = sum(1 for row in self.state.rows if row.exit_code is not None)
        failures = sum(
            1
            for row in self.state.rows
            if row.exit_code is not None and row.bucket == "fail"
        )
        bad_shards = [job for job in self.jobs if job.exit_code not in (0, None)]
        suffix = "" if not bad_shards else f"，{len(bad_shards)} 個分片非零退出"
        if self.cancelling:
            suffix += "（已中止）"
        self.current_case_label.setText(
            f"完成：{finished}/{self.state.total} cases，{failures} 個未通過{suffix}"
        )
        self.statusBar().showMessage(f"Run finished · {self.run_dir}")
        self.last_phase = "run"
        self.last_exit_code = max(
            (job.exit_code or 0) for job in self.jobs
        ) if self.jobs else None
        self._finish_idle()

    def _merge_summaries(self) -> None:
        """Section 4.4: one summary.tsv out of every shard's own file."""
        if self.run_dir is None:
            return
        merged = self.run_dir / "summary.tsv"
        header = "case\tstatus\texit_code\tduration_ms\tcase_dir\n"
        rows: list[str] = []
        for job in self.jobs:
            path = job.summary_path or (job.output_dir / "summary.tsv")
            if not Path(path).is_file():
                continue
            for line in Path(path).read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                if not line or line.startswith("case\t"):
                    continue
                rows.append(line)
        try:
            merged.write_text(header + "\n".join(rows) + ("\n" if rows else ""),
                              encoding="utf-8")
            self.state.summary_path = merged
            self._append_log(f"--- merged {len(rows)} row(s) into {merged} ---")
        except OSError as error:
            self._append_log(f"--- could not write merged summary: {error} ---")

    def _write_stats(self) -> None:
        """Section 4.5, written to disk so it survives closing the window."""
        if self.run_dir is None:
            return
        finished = [row for row in self.state.rows if row.exit_code is not None]
        status_tally: dict[str, int] = {}
        for row in finished:
            status_tally[row.status] = status_tally.get(row.status, 0) + 1
        lines = [f"# {self.state.plan.description if self.state.plan else ''}", ""]
        lines.append("[status]")
        for status, count in sorted(status_tally.items(), key=lambda i: -i[1]):
            lines.append(f"{status:<18} {count}")
        lines.append("")
        lines.append("[per group]")
        counts: dict[int, dict[str, int]] = {}
        for row in finished:
            entry = counts.setdefault(
                row.group_number, {"pass": 0, "fail": 0, "skip": 0, "warn": 0}
            )
            entry[row.bucket] += 1
        for number in sorted(counts):
            group = GROUPS_BY_NUMBER.get(number)
            entry = counts[number]
            done = sum(entry.values())
            rate = f"{100.0 * entry['pass'] / done:.1f}%" if done else "-"
            lines.append(
                f"{number:>2} {(group.label if group else '(ungrouped)'):<32} "
                f"n={done:<6} pass={entry['pass']:<6} fail={entry['fail']:<6} "
                f"skip={entry['skip']:<6} warn={entry['warn']:<5} {rate}"
            )
        lines.append("")
        lines.append("[shards]")
        for job in self.jobs:
            lines.append(
                f"{job.tag:<28} cases={len(job.cases):<6} exit={job.exit_code}"
            )
        try:
            (self.run_dir / "stats.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
        except OSError:
            pass

    def _set_busy(self, busy: bool) -> None:
        for button in (
            self.run_button,
            self.run_stop_button,
            self.run_skip_passed_button,
            self.check_button,
            self.discover_button,
            self.preview_button,
        ):
            button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)

    def _finish_idle(self) -> None:
        self.phase = "idle"
        self._set_busy(False)

    # --------------------------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------------------------
    @staticmethod
    def _tail(text: str, lines: int) -> str:
        rows = text.splitlines()
        if len(rows) <= lines:
            return "\n".join(rows)
        return "\n".join(
            [f"... ({len(rows) - lines} earlier lines omitted)"] + rows[-lines:]
        )

    @staticmethod
    def _read_tail(path: Path, lines: int, byte_cap: int = 262_144) -> str:
        try:
            size = path.stat().st_size
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                if size > byte_cap:
                    handle.seek(size - byte_cap)
                    handle.readline()
                return MainWindow._tail(handle.read(), lines)
        except OSError as error:
            return f"<unreadable: {error}>"

    def _case_detail(self, row: CaseRow) -> list[str]:
        out = [
            f"-- [{row.index}] {row.case_name}",
            f"   status={row.status} exit={row.exit_code} "
            f"duration={format_duration(row.duration_ms)} shard={row.job_tag or '-'}",
        ]
        group = GROUPS_BY_NUMBER.get(row.group_number)
        if group is not None:
            out.append(f"   group: 第 {group.number} 組 {group.label}")
        if row.case_dir is None or not row.case_dir.is_dir():
            out.append("   case dir: <missing>")
            return out
        out.append(f"   case dir: {row.case_dir}")
        qpa = row.qpa or (row.case_dir / "results.qpa")
        if Path(qpa).is_file():
            code, detail = qpa_reason(Path(qpa))
            if code or detail:
                out.append(f"   qpa result: {code} · {detail[:300]}")
        command = row.case_dir / "command.txt"
        if not command.is_file():
            command = row.case_dir / "driver-command.txt"
        if command.is_file():
            out.append("   command:")
            out += [f"     {line}" for line in self._read_tail(command, 12).splitlines()]
        jsonl = row.case_dir / "systemc.jsonl"
        if jsonl.is_file():
            interesting: list[str] = []
            for line in self._read_tail(jsonl, 400).splitlines():
                if any(
                    mark in line
                    for mark in ('"type":"done"', '"type":"error"', '"error"', '"warning"')
                ):
                    interesting.append(line[:600])
            if interesting:
                out.append("   systemc.jsonl (done/error lines):")
                out += [f"     {line}" for line in interesting[-6:]]
            else:
                out.append("   systemc.jsonl: no done/error line")
        for name in ("stderr.log", "stdout.log", "case.log", "run.log"):
            candidate = row.case_dir / name
            if candidate.is_file() and candidate.stat().st_size:
                out.append(f"   {name} (tail):")
                out += [
                    f"     {line}" for line in self._read_tail(candidate, 25).splitlines()
                ]
        pngs = sorted((row.case_dir / "systemc").glob("*.png"))
        if pngs:
            out.append(f"   systemc png: {len(pngs)} file(s), e.g. {pngs[0].name}")
        return out

    def build_diagnostics(self, only_first_fail: bool = False) -> str:
        """One self-contained plain-text report: which tier was asked for, what
        the plan came to, what the wiring resolved to, and the detail behind
        each failure.

        With only_first_fail, the problem-case sections are trimmed down to
        just the first case whose bucket is "fail" -- a quick paste for one
        bug report instead of a full-run dump."""
        now = datetime.now().astimezone()
        lines: list[str] = []
        add = lines.append

        add("=== PvrGPU dEQP · 四層回歸 — diagnostics ==="
            + (" (first Fail only)" if only_first_fail else ""))
        add(f"generated : {now.isoformat(timespec='seconds')}")
        add(f"ui        : {Path(__file__).resolve()}")
        add(f"doc       : {TIER_DOC}")
        add(f"host      : {platform.platform()} ({platform.machine()})")
        add(
            f"python    : {sys.version.split()[0]}  Qt/PySide6: "
            f"{getattr(__import__('PySide6'), '__version__', '?')}"
        )
        add(f"repo      : {REPO_ROOT}")
        add("")

        mode = self.mode_combo.currentText()
        add("[selection]")
        add(f"mode          : {mode}")
        if mode in (MODE_TIER, MODE_GROUP):
            tier = self._current_tier()
            add(f"tier          : {tier.label} · {tier.rule}")
        if mode == MODE_GROUP:
            add(f"group         : {self.group_combo.currentText()}")
        elif mode == MODE_CUSTOM:
            add(f"custom case   : {self.custom_case_edit.text().strip() or '<blank>'}")
        elif mode == MODE_CASELIST:
            add(f"caselist file : {self.caselist_edit.text().strip() or '<blank>'}")
        add(f"shards        : {self._effective_shards()}")
        add(f"log images    : {self._effective_log_images()}")
        add(f"on failure    : {'stop' if self.stop_on_fail else 'keep going'}")
        add("")

        plan = self.state.plan
        add("[plan]")
        if plan is None:
            add("<no plan has been run in this window yet>")
        else:
            add(f"name          : {plan.name}")
            add(f"description   : {plan.description}")
            add(f"total cases   : {plan.total_cases}")
            for bucket in plan.buckets:
                add(
                    f"  bucket {bucket.tag:<22} {len(bucket.cases):>6} cases "
                    f"{bucket.gl_config}"
                    + (f"  ({bucket.reason})" if bucket.reason else "")
                )
            drifted = [
                entry for entry in plan.groups if entry.selected and entry.drift
            ]
            if drifted:
                add("  discovery drift vs docs/dEQP_4level.md:")
                for entry in drifted:
                    add(
                        f"    第 {entry.group.number} 組 {entry.group.label}: "
                        f"discovery {entry.discovered}, 文件 {entry.group.total}"
                    )
        add("")

        add("[paths as typed in the UI]  (blank = taken from config/local.env)")
        for caption, widget in (
            ("PCO driver (Mesa prefix)", getattr(self, "mesa_edit", None)),
            ("SystemC bridge", getattr(self, "bridge_edit", None)),
            ("dEQP binary", getattr(self, "deqp_binary_edit", None)),
            ("dEQP build dir", getattr(self, "deqp_build_edit", None)),
            ("Output root", getattr(self, "output_edit", None)),
        ):
            if widget is not None:
                add(f"{caption:<26}: {widget.text().strip() or '<blank>'}")
        add(f"{'discovery dir':<26}: {self._discovery_dir()}")
        for module in DISCOVERY_MODULES:
            cases = self.discovery.get(module)
            add(f"{'  ' + module:<26}: {len(cases) if cases else '<missing>'}")
        add("")

        add("[wiring the runner reported]")
        for caption, value in (
            ("Module", self.state.module),
            ("dEQP runner", self.state.runner),
            ("Archive dir", self.state.archive_dir),
            ("PCO driver (Mesa)", self.state.mesa_prefix),
            ("SystemC bridge", self.state.systemc_lib),
            ("Host arch", self.state.host_arch),
        ):
            add(f"{caption:<26}: {value or '<not reported>'}")
        add("")

        add("[shards]")
        if not self.jobs:
            add("<none>")
        for job in self.jobs:
            add(
                f"{job.tag:<28} module={job.module:<7} config={job.gl_config:<18} "
                f"cases={len(job.cases):<6} exit={job.exit_code} dir={job.output_dir}"
            )
        add("")

        add("[run]")
        add(f"run dir       : {self.run_dir or '<none>'}")
        add(f"summary       : {self.state.summary_path or '<none>'}")
        started = self.state.started_at
        add(
            f"started       : "
            f"{started.isoformat(timespec='seconds') if started else '<none>'}"
        )
        add(f"elapsed       : {format_duration(self.state.duration_ms)}")
        add(f"last phase    : {self.last_phase or '<none>'}  exit={self.last_exit_code}")
        add("")

        finished = [row for row in self.state.rows if row.exit_code is not None]
        buckets = {"pass": 0, "fail": 0, "skip": 0, "warn": 0}
        for row in finished:
            buckets[row.bucket] += 1
        add("[totals]")
        add(
            f"total={self.state.total} finished={len(finished)} "
            f"pass={buckets['pass']} fail={buckets['fail']} "
            f"skip={buckets['skip']} warn={buckets['warn']}"
        )
        add("")

        # Notable = anything worth a human looking at: real failures/warnings,
        # plus NotSupported (unlike Waiver, that often means a real capability
        # gap rather than an intentionally accepted exclusion).
        def _is_notable(row: CaseRow) -> bool:
            return (
                row.bucket in ("fail", "warn")
                or row.status.strip().casefold() == "notsupported"
            )

        if only_first_fail:
            notable = [row for row in self.state.rows if row.bucket == "fail"][:1]
        else:
            notable = [row for row in self.state.rows if _is_notable(row)]
        add(
            f"[problem cases]  ({len(notable)} of {len(self.state.rows)};"
            f" first {min(len(notable), DIAGNOSTICS_MAX_SUMMARY_ROWS)} shown)"
        )
        add(
            f"{'#':>5}  {'status':<13} {'exit':>4} {'duration':>9}  "
            f"{'shard':<18} case"
        )
        if not notable:
            add("(no Fail case found)" if only_first_fail
                else "(none -- every case passed or was an accepted waiver)")
        for row in notable[:DIAGNOSTICS_MAX_SUMMARY_ROWS]:
            add(
                f"{row.index:>5}  {row.status:<13} "
                f"{('' if row.exit_code is None else row.exit_code):>4} "
                f"{format_duration(row.duration_ms):>9}  "
                f"{row.job_tag:<18} {row.case_name}"
            )
        if len(notable) > DIAGNOSTICS_MAX_SUMMARY_ROWS:
            add(f"... ({len(notable) - DIAGNOSTICS_MAX_SUMMARY_ROWS} more not shown)")
        add("")

        add(
            f"[problem cases in detail]  ({len(notable)} total;"
            f" first {min(len(notable), DIAGNOSTICS_MAX_CASES)} shown)"
        )
        if not notable:
            add("(none)")
        for row in notable[:DIAGNOSTICS_MAX_CASES]:
            lines.extend(self._case_detail(row))
            add("")

        if notable and not only_first_fail:
            add("[UI log tail]")
            add(self._tail(self.log_view.toPlainText(), DIAGNOSTICS_LOG_LINES))
            add("")
        add("=== end of diagnostics ===")
        return "\n".join(lines)

    def copy_diagnostics(self, only_first_fail: bool = False) -> None:
        try:
            report = self.build_diagnostics(only_first_fail=only_first_fail)
        except Exception as error:  # noqa: BLE001 - a diagnostic must not crash the UI
            QMessageBox.critical(self, "PvrGPU", f"產生診斷資訊失敗：{error}")
            return

        if only_first_fail and not any(row.bucket == "fail" for row in self.state.rows):
            self.statusBar().showMessage("目前沒有 Fail 的 case，沒有東西可以複製", 8000)
            return

        saved: Path | None = None
        filename = "diagnostics-first-fail.txt" if only_first_fail else "diagnostics.txt"
        if self.run_dir:
            candidate = Path(self.run_dir) / filename
            try:
                candidate.parent.mkdir(parents=True, exist_ok=True)
                candidate.write_text(report, encoding="utf-8")
                saved = candidate
            except OSError:
                saved = None

        if saved is not None:
            report = f"diagnostics file: {saved}\n\n{report}"
        QApplication.clipboard().setText(report)

        label = "第一個 Fail 的診斷資訊" if only_first_fail else "診斷資訊"
        message = f"{label}已複製（{format_bytes(len(report.encode('utf-8')))}）"
        if saved is not None:
            message += f" · 也寫到 {saved}"
        self.statusBar().showMessage(message, 8000)

    # --------------------------------------------------------------------------
    # Opening things
    # --------------------------------------------------------------------------
    def _open_path(self, path: Path | None) -> None:
        if path is None or not Path(path).exists():
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _open_output_root(self) -> None:
        self._open_path(self._output_root())

    def _open_run_dir(self) -> None:
        self._open_path(self.run_dir)

    def _open_summary(self) -> None:
        self._open_path(self.state.summary_path)

    def _open_case_item(self, item: QTableWidgetItem) -> None:
        name_item = self.results_table.item(item.row(), 1)
        if name_item is None:
            return
        directory = name_item.data(Qt.ItemDataRole.UserRole)
        if directory:
            self._open_path(Path(directory))

    def _open_artifact_item(self, item: QTableWidgetItem) -> None:
        path_item = self.artifacts_table.item(item.row(), 0)
        if path_item is None:
            return
        value = path_item.data(Qt.ItemDataRole.UserRole)
        if value:
            self._open_path(Path(value))

    # --------------------------------------------------------------------------
    # Settings
    # --------------------------------------------------------------------------
    def _path_widgets(self) -> dict[str, QLineEdit]:
        return {
            "mesa": self.mesa_edit,
            "bridge": self.bridge_edit,
            "deqp_binary": self.deqp_binary_edit,
            "deqp_build": self.deqp_build_edit,
            "output": self.output_edit,
            "caselist": self.caselist_edit,
        }

    def _restore_settings(self) -> None:
        for key, widget in self._path_widgets().items():
            stored = self.settings.value(f"paths/{key}", "", str)
            if stored:
                widget.setText(stored)
        mode = self.settings.value("selection/mode", MODE_TIER, str)
        index = self.mode_combo.findText(mode)
        if index >= 0:
            self.mode_combo.setCurrentIndex(index)
        tier_id = self.settings.value("selection/tier", "L1", str)
        index = self.tier_combo.findData(tier_id)
        if index >= 0:
            self.tier_combo.setCurrentIndex(index)
        try:
            number = int(self.settings.value("selection/group", 1))
        except (TypeError, ValueError):
            number = 1
        index = self.group_combo.findData(number)
        if index >= 0:
            self.group_combo.setCurrentIndex(index)
        try:
            self.shards_spin.setValue(int(self.settings.value("run/shards", 8)))
        except (TypeError, ValueError):
            self.shards_spin.setValue(8)
        self._update_tier_note()
        self._update_group_note()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self.aux_process is not None or self.active_jobs:
            answer = QMessageBox.question(
                self,
                "PvrGPU",
                "還有工作正在執行，要中止並關閉嗎？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        for key, widget in self._path_widgets().items():
            self.settings.setValue(f"paths/{key}", widget.text())
        self._sync_selection_to_settings()
        if self.aux_process is not None:
            self.aux_process.kill()
            self.aux_process.waitForFinished(2000)
        for process in list(self.active_jobs):
            process.kill()
            process.waitForFinished(2000)
        super().closeEvent(event)

    # --------------------------------------------------------------------------
    # Style
    # --------------------------------------------------------------------------
    def _apply_style(self) -> None:
        theme = self.theme
        self.setStyleSheet(
            f"""
            QWidget {{ color: {theme.text}; font-size: 13px; }}
            QMainWindow, QScrollArea#sideScroll, QWidget#sidePanel,
            QWidget#sideActions {{
                background: {theme.window};
            }}
            QWidget#sideActions {{ border-top: 1px solid {theme.border}; }}
            QGroupBox {{
                border: 1px solid {theme.border};
                border-radius: 10px;
                margin-top: 16px;
                padding: 14px 12px 12px 12px;
                background: {theme.surface};
                font-weight: 600;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0 6px;
                background: {theme.surface};
                color: {theme.muted};
            }}
            QLabel#caption {{ color: {theme.muted}; font-size: 12px; }}
            QLabel#note {{ color: {theme.muted}; font-size: 12px; }}
            QLabel#planNote {{
                color: {theme.text};
                font-size: 12px;
                background: {theme.subtle};
                border: 1px solid {theme.border};
                border-radius: 6px;
                padding: 6px 8px;
            }}
            QFrame#header {{
                background: {theme.surface};
                border: 1px solid {theme.border};
                border-radius: 10px;
            }}
            QLabel#currentCase, QLabel#dashboardTitle {{
                font-size: 15px; font-weight: 600; color: {theme.text};
            }}
            QLabel[tile="true"] {{
                background: {theme.subtle};
                border: 1px solid {theme.border};
                border-radius: 8px;
                padding: 8px 4px;
                font-weight: 600;
                color: {theme.text};
            }}
            QLabel#tile-pass {{ background: {theme.ok_bg}; color: {theme.ok}; }}
            QLabel#tile-fail {{ background: {theme.bad_bg}; color: {theme.bad}; }}
            QLabel#tile-warn {{ background: {theme.warn_bg}; color: {theme.warn}; }}
            QLineEdit, QComboBox, QSpinBox, QAbstractSpinBox {{
                background: {theme.field};
                border: 1px solid {theme.border};
                border-radius: 6px;
                padding: 5px 8px;
                min-height: 20px;
                color: {theme.text};
                selection-background-color: {theme.accent};
                selection-color: {theme.accent_text};
            }}
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{
                border-color: {theme.accent};
            }}
            QComboBox::drop-down {{ border: none; width: 20px; }}
            QComboBox QAbstractItemView {{
                background: {theme.surface};
                color: {theme.text};
                border: 1px solid {theme.border};
                selection-background-color: {theme.accent};
                selection-color: {theme.accent_text};
            }}
            QCheckBox {{ color: {theme.text}; spacing: 7px; }}
            QPushButton {{
                padding: 6px 12px;
                border: 1px solid {theme.border};
                border-radius: 7px;
                background: {theme.surface};
                color: {theme.text};
            }}
            QPushButton:hover {{ background: {theme.subtle}; }}
            QPushButton#primary {{
                background: {theme.accent};
                border-color: {theme.accent};
                color: {theme.accent_text};
                font-weight: 600;
            }}
            QPushButton#quit {{ color: {theme.muted}; }}
            QPushButton#quit:hover {{
                color: {theme.bad}; border-color: {theme.bad};
            }}
            QPushButton:disabled {{ color: {theme.muted}; background: {theme.subtle}; }}
            QTableWidget {{
                background: {theme.surface};
                alternate-background-color: {theme.subtle};
                border: 1px solid {theme.border};
                border-radius: 8px;
                gridline-color: {theme.border};
                color: {theme.text};
            }}
            QHeaderView::section {{
                background: {theme.subtle};
                border: none;
                border-bottom: 1px solid {theme.border};
                padding: 6px;
                font-weight: 600;
                color: {theme.muted};
            }}
            QTabWidget::pane {{
                border: 1px solid {theme.border};
                background: {theme.surface};
                border-radius: 8px;
            }}
            QTabBar::tab {{
                padding: 6px 14px;
                margin-right: 2px;
                border: 1px solid transparent;
                border-radius: 7px;
                color: {theme.muted};
            }}
            QTabBar::tab:selected {{
                background: {theme.accent};
                color: {theme.accent_text};
                font-weight: 600;
            }}
            QProgressBar {{
                border: 1px solid {theme.border};
                border-radius: 7px;
                background: {theme.subtle};
                height: 18px;
                text-align: center;
                color: {theme.text};
            }}
            QProgressBar::chunk {{ background: {theme.accent}; border-radius: 6px; }}
            QProgressBar#bar-pass::chunk {{ background: {theme.ok}; }}
            QProgressBar#bar-fail::chunk {{ background: {theme.bad}; }}
            QProgressBar#bar-skip::chunk {{ background: {theme.skip}; }}
            QProgressBar#bar-warn::chunk {{ background: {theme.warn}; }}
            QPlainTextEdit {{
                background: {theme.log_bg};
                color: {theme.log_text};
                border: 1px solid {theme.border};
                border-radius: 8px;
            }}
            QSplitter::handle {{ background: {theme.border}; width: 1px; }}
            QStatusBar {{ color: {theme.muted}; }}
            QMenuBar {{ background: {theme.window}; color: {theme.text}; }}
            QMenuBar::item:selected {{
                background: {theme.accent}; color: {theme.accent_text};
            }}
            QMenu {{
                background: {theme.surface}; color: {theme.text};
                border: 1px solid {theme.border};
            }}
            QMenu::item:selected {{
                background: {theme.accent}; color: {theme.accent_text};
            }}
            """
        )


def main() -> int:
    application = QApplication(sys.argv)
    application.setApplicationName("PvrGPU dEQP 四層回歸")

    system_theme = detect_theme(application)
    preference = QSettings("PvrGPU", "deqp-4level-ui").value(
        "appearance/theme", DEFAULT_APPEARANCE, str
    )
    if preference not in ("system", "light", "dark"):
        preference = DEFAULT_APPEARANCE
    theme = resolve_theme(preference, system_theme)

    application.setStyle("Fusion")
    apply_palette(application, theme)
    window = MainWindow(theme, system_theme=system_theme, appearance=preference)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
