"""Shared catalog and deterministic planning for the four-tier dEQP regression.

Both the desktop UI and command-line runner use this module so tier quotas,
long-case exclusions, sampling, sharding, and framebuffer configs stay aligned.
It has no UI or third-party dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

__all__ = [
    "PASS_STATUSES",
    "SKIP_STATUSES",
    "WARNING_STATUSES",
    "EXACT_CASE_RE",
    "SUITE_TO_MODULE",
    "DEFAULT_GL_CONFIG",
    "MS4_GL_CONFIG",
    "CONFIG_OVERRIDES",
    "LONG_ONLY_PREFIXES",
    "LONG_ONLY_DOCUMENTED_BY_GROUP",
    "is_long_only_case",
    "AVERAGE_SECONDS_PER_CASE",
    "Group4",
    "GROUPS_4LEVEL",
    "GROUPS_BY_NUMBER",
    "Tier",
    "TIERS",
    "TIERS_BY_ID",
    "L4_MODULES",
    "DISCOVERY_MODULES",
    "sample_evenly",
    "round_robin_shards",
    "gl_config_for",
    "module_for_case",
    "group_for_case",
    "GroupPlan",
    "Bucket",
    "RunPlan",
    "bucket_case",
    "plan_from_cases",
    "build_plan",
]

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

# These families are intentionally L4-only.  They have either unbounded shader
# loops or repeatedly measured multi-minute execution on the SystemC model.
# Keeping the policy as case prefixes makes it deterministic and reviewable:
# prior timing files are evidence for changing this list, never an input that
# silently changes a future plan.  Explicit one-case/caselist debug runs are
# not filtered; only the named L1/L2/L3 tier plans apply this policy.
LONG_ONLY_PREFIXES: tuple[str, ...] = (
    "dEQP-GLES3.stress.long_shaders.",
    "dEQP-GLES3.stress.long_running_shaders.",
    "dEQP-GLES3.functional.multisample.default_framebuffer.constancy_",
    "dEQP-GLES3.functional.multisample.fbo_4_samples.constancy_",
    "dEQP-GLES3.functional.multisample.fbo_8_samples.constancy_",
    "dEQP-GLES3.functional.multisample.fbo_max_samples.constancy_",
    "dEQP-GLES31.functional.multisample.default_framebuffer.constancy_",
    "dEQP-GLES31.functional.texture.multisample.samples_8.sample_mask_",
    "dEQP-GLES31.functional.draw_indirect.compute_interop.large.",
    "dEQP-GLES31.stress.draw_indirect.drawarrays.data_over_bounds_with_primcount",
    "dEQP-GLES31.stress.draw_indirect.drawelements.data_over_bounds_with_primcount",
    "dEQP-GLES31.functional.image_load_store.2d_array.atomic.",
    "dEQP-GLES31.functional.image_load_store.3d.atomic.",
    "dEQP-GLES31.functional.image_load_store.cube.atomic.",
    "dEQP-GLES31.functional.synchronization.inter_call.with_memory_barrier.",
    "dEQP-GLES31.functional.synchronization.inter_call.without_memory_barrier.image_atomic_dispatch_100_calls_128x128_invocations",
    "dEQP-GLES31.functional.synchronization.inter_call.without_memory_barrier.ssbo_atomic_dispatch_100_calls_32k_invocations",
)

# Counts in the documented 30-group catalog.  Import-time validation catches a
# quota/table edit that forgets to update the published tier totals.
LONG_ONLY_DOCUMENTED_BY_GROUP: dict[int, int] = {
    18: 40,
    20: 66,
    22: 9,
    28: 102,
    29: 30,
    30: 20,
}


def is_long_only_case(case_name: str) -> bool:
    """Return whether a case belongs only in L4 (or an explicit debug run)."""
    return any(case_name.startswith(prefix) for prefix in LONG_ONLY_PREFIXES)

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
        if tier_id == "L3":
            return self.total - LONG_ONLY_DOCUMENTED_BY_GROUP.get(self.number, 0)
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
                      "stress.long_running_shaders."), 40, 0, 0,
           note="40 條皆為 L4-only 長測項；L1/L2/L3 不執行"),
    Group4(19, "Compute", _GLES31,
           _selectors(_GLES31, "functional.compute."), 195, 100, 195,
           note="子群 basic 41、shared_var 136、indirect_dispatch 18；"
                "與第 21 組 SSBO 共用 shader 寫記憶體的路徑"),
    Group4(20, "Draw indirect", _GLES31,
           _selectors(_GLES31,
                      "functional.draw_indirect.",
                      "stress.draw_indirect."), 244, 100, 178,
           note="64 條 large compute_interop 與 2 條長時間 stress 為 L4-only；"
                "其餘 178 條在 L2 完整覆蓋"),
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
           note="4 條 default_framebuffer.constancy 與 5 條 8× texture sample-mask "
                "為 L4-only；"
                "先跑第 30 組（基本 MSAA 光柵化）再跑這組；"
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
           _selectors(_GLES31, "functional.image_load_store."), 747, 100, 400,
           note="102 條 2d_array/3d/cube atomic 為 L4-only；"
                "L1/L2 從其餘 645 條等距取樣"),
    Group4(29, "Sync + atomic counter", _GLES31,
           _selectors(_GLES31,
                      "functional.synchronization.",
                      "functional.atomic_counter."), 402, 100, 372,
           note="28 條 with_memory_barrier 與 2 條超大 atomic dispatch "
                "為 L4-only；其餘 372 條在 L2 完整覆蓋"),
    Group4(30, "Basic MSAA", _GLES3,
           _selectors(_GLES3, "functional.multisample."), 64, 44, 44,
           note="20 條 constancy（default／4×／8×／max）為 L4-only；"
                "default_framebuffer 子群（16 條）需要 ms4"),
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
    Tier("L1", "L1 · Very Fast Regression", 2474, "20 min", "2.7 min", "5 min",
         "每次 commit 前", 8, "disable",
         "排除 L4-only 長測項後，每組 min(N, 100) 條；第 11 組 compressed 蓋在 30",
         "涵蓋 29 個一般組；第 18 組全為長測項，只在 L4 執行。"),
    Tier("L2", "L2 · Fast Regression", 7750, "54 min", "8.6 min", "15 min",
         "修完一個 subsystem", 8, "disable",
         "排除 L4-only 長測項後，每組 min(N, 400) 條；第 11 組 compressed 蓋在 120",
         "完整跑一般 compute／draw indirect；large compute-interoperability 只在 L4。"),
    Tier("L3", "L3 · Detail Regression", 30224, "4.0 h", "40 min", "1 h",
         "milestone、過夜", 8, "disable",
         "30 組目錄扣除 L4-only 長測項後全跑",
         "只有 L3 跑得完一般測項的是第 5、6、9、10、12、13、14、15、21、22、24、25、"
         "26、27、28 組。"),
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
        if not 0 <= group.l1 <= group.l2 <= group.total:
            raise RuntimeError(
                f"group {group.number} quotas must satisfy 0 <= L1 <= L2 <= N"
            )
    def eligible(group: Group4) -> int:
        return group.total - LONG_ONLY_DOCUMENTED_BY_GROUP.get(group.number, 0)

    for number, excluded in LONG_ONLY_DOCUMENTED_BY_GROUP.items():
        if number not in GROUPS_BY_NUMBER or not 0 < excluded <= GROUPS_BY_NUMBER[number].total:
            raise RuntimeError(f"invalid L4-only documented count for group {number}")
    for group in GROUPS_4LEVEL:
        if group.l2 > eligible(group):
            raise RuntimeError(
                f"group {group.number} L2 quota exceeds its non-long case count"
            )

    totals = {
        "L1": sum(g.l1 for g in GROUPS_4LEVEL),
        "L2": sum(g.l2 for g in GROUPS_4LEVEL),
        "L3": sum(eligible(g) for g in GROUPS_4LEVEL),
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
    long_excluded: int = 0

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

    @property
    def long_excluded_count(self) -> int:
        return sum(entry.long_excluded for entry in self.groups)

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
        unfiltered_pool = [
            case_name
            for case_name in discovery.get(group.module, ())
            if group.matches(case_name)
        ]
        long_excluded = 0 if tier.whole_modules else sum(
            is_long_only_case(case_name) for case_name in unfiltered_pool
        )
        pool = (
            unfiltered_pool
            if tier.whole_modules
            else [case_name for case_name in unfiltered_pool if not is_long_only_case(case_name)]
        )
        quota = group.quota(tier.id)
        chosen = [name for name in sample_evenly(pool, quota) if name not in seen]
        seen.update(chosen)
        plan.groups.append(
            GroupPlan(
                group=group,
                discovered=len(unfiltered_pool),
                quota=quota,
                selected=len(chosen),
                long_excluded=long_excluded,
            )
        )
        for case_name in chosen:
            add(case_name, group.module)

    plan.buckets = sorted(
        buckets.values(), key=lambda bucket: (bucket.module, bucket.gl_config)
    )
    long_note = (
        f"{plan.long_excluded_count} long cases 留在 L4 · "
        if plan.long_excluded_count else ""
    )
    plan.description = (
        f"{tier.id} · {len(groups)} 組 · {plan.total_cases} cases · "
        f"{long_note}{plan.shards} shard"
    )
    return plan
