#!/usr/bin/env python3
"""Run the live dEQP-against-pvrgpu regression across every catalogued group.

This sits one layer above ``deqp_group_runner.py``: that script already
drives ``pvrgpu-deqp`` (dEQP linked directly against the Mesa/PCO driver and
the SystemC bridge -- no RenderDoc capture/replay involved) one catalogued
group at a time, spawning a fresh process per exact case because the current
SystemC bridge defers simulation until process exit and only retains its
latest submitted command.

This script adds the piece that was still manual: looping over every group in
the catalog (``deqp_groups.GROUP_SPECS``), consolidating a single pass/fail
report across all of them (JSON + Markdown), and -- for every case that comes
back failed -- building a self-contained debug bundle: the artifacts the
original run already produced (driver command/counter, SystemC JSONL/PNG),
plus an isolated re-run of that one case with images forced on.

Usage:
    python3 tools/deqp_pvrgpu_regression.py \
        --runner "$PVRGPU_BUILD_DIR/bin/pvrgpu-deqp" \
        --output-dir outputs/deqp_regression/$(date +%Y%m%d_%H%M%S)

    # also discovery-list the blocked GLES3/3.1/3.2 groups, to track whether
    # their live case counts have drifted from the catalogued ones:
        --groups all --include-blocked

Prefer the ``run_deqp_regression.sh`` wrapper at the repo root, which sources
config/local.env and fills in --runner/--systemc-api-lib automatically.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deqp_groups import GroupSpec, GROUP_SPECS, get_group  # noqa: E402

SCHEMA = "pvrgpu.deqp-regression-report.v1"
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]")
PASS_STATUSES = {"pass"}
SKIP_STATUSES = {"notsupported", "waiver"}
WARNING_STATUSES = {"qualitywarning", "compatibilitywarning"}
SYSTEMC_ARTIFACT_NAMES = ("systemc.jsonl", "driver-command.txt", "driver-counter.txt")
DASHBOARD_SCHEMA = "pvrgpu.deqp-dashboard-state.v1"
MAX_HISTORY_PER_GROUP = 30
MAX_RUN_LOG = 50


def safe_case_name(case_name: str) -> str:
    """Mirror deqp_live_ui.safe_case_name / the native runner's directory mapping."""
    return SAFE_NAME_RE.sub("_", case_name) or "unnamed"


def status_bucket(status: str) -> str:
    normalized = status.casefold()
    if normalized in PASS_STATUSES:
        return "passed"
    if normalized in SKIP_STATUSES:
        return "skipped"
    if normalized in WARNING_STATUSES:
        return "warnings"
    return "failed"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runner", required=True, type=Path, help="path to the pvrgpu-deqp binary")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--groups",
        default="available",
        help="comma-separated group ids, or 'available' (default: only groups "
        "the current driver can execute) or 'all'",
    )
    parser.add_argument(
        "--include-blocked",
        action="store_true",
        help="also run --list-only discovery on blocked groups, to track "
        "whether their live case count has drifted from the catalogued one",
    )
    parser.add_argument("--max-cases-per-group", type=int, default=0)
    parser.add_argument("--surface", default="pbuffer")
    parser.add_argument("--watchdog", default="disable")
    parser.add_argument("--log-images", choices=("enable", "disable"), default="disable")
    parser.add_argument("--gl-config", default="")
    parser.add_argument("--systemc-api-lib", default="")
    parser.add_argument(
        "--no-debug-bundle",
        action="store_true",
        help="skip building a per-failure debug bundle (isolated re-run + artifact copy)",
    )
    parser.add_argument(
        "--fail-on-blocked-drift",
        action="store_true",
        help="exit non-zero if a blocked group's live discovered case count "
        "differs from deqp_groups.GROUP_SPECS.locked_case_count",
    )
    parser.add_argument(
        "--dashboard-dir",
        type=Path,
        default=None,
        help="cumulative dashboard location (default: a 'dashboard' directory "
        "next to --output-dir's parent, shared across every run so history "
        "accumulates instead of resetting per run)",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="skip updating the cumulative dashboard.html for this run",
    )
    return parser.parse_args(argv)


def select_groups(spec_arg: str) -> list[GroupSpec]:
    if spec_arg == "available":
        return [group for group in GROUP_SPECS if group.available]
    if spec_arg == "all":
        return list(GROUP_SPECS)
    ids = [item.strip() for item in spec_arg.split(",") if item.strip()]
    return [get_group(item) for item in ids]


def group_runner_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "deqp_group_runner.py"),
        f"--runner={args.runner}",
        f"--output-dir={output_dir}",
        f"--surface={args.surface}",
        f"--watchdog={args.watchdog}",
        f"--log-images={args.log_images}",
    ]
    if args.gl_config:
        command.append(f"--gl-config={args.gl_config}")
    if args.systemc_api_lib:
        command.append(f"--systemc-api-lib={args.systemc_api_lib}")
    return command


def run_group(args: argparse.Namespace, group: GroupSpec, output_dir: Path, *, list_only: bool) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [*group_runner_command(args, output_dir), f"--group-id={group.id}"]
    if list_only:
        command.append("--list-only")
    elif args.max_cases_per_group > 0:
        command.append(f"--max-cases={args.max_cases_per_group}")

    print(f"==> {group.id}: {group.label} ({'discovery only' if list_only else 'executing'})", flush=True)
    started = time.monotonic()
    completed = subprocess.run(command, text=True, check=False)
    duration = time.monotonic() - started

    case_list_path = output_dir / "case-list.txt"
    live_case_count = 0
    if case_list_path.is_file():
        live_case_count = sum(
            1 for line in case_list_path.read_text(encoding="utf-8").splitlines() if line.strip()
        )

    result: dict = {
        "group_id": group.id,
        "label": group.label,
        "suite": group.suite,
        "output_dir": str(output_dir),
        "available": group.available,
        "availability_reason": group.availability_reason,
        "list_only": list_only,
        "runner_exit": completed.returncode,
        "duration_seconds": round(duration, 3),
        "locked_case_count": group.locked_case_count,
        "live_case_count": live_case_count,
        "case_count_drift": live_case_count - group.locked_case_count,
    }

    summary_path = output_dir / "batch-summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            summary = {}
        for key in ("total", "completed", "passed", "skipped", "warnings", "failed", "cancelled"):
            if key in summary:
                result[key] = summary[key]
    return result


def load_failed_cases(group_output_dir: Path) -> list[dict]:
    journal_path = group_output_dir / "batch-results.jsonl"
    if not journal_path.is_file():
        return []
    failed = []
    for line in journal_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if status_bucket(record.get("status", "")) == "failed":
            failed.append(record)
    return failed


def build_debug_bundle(
    args: argparse.Namespace,
    group: GroupSpec,
    group_output_dir: Path,
    case_record: dict,
    debug_root: Path,
) -> Path:
    """Collect what's already known about a failure, then re-run it in isolation.

    The isolated re-run always forces --deqp-log-images=enable, so a debug
    bundle has PNG evidence even when the batch run itself had images off.
    """
    case_name = case_record["case"]
    case_dir = group_output_dir / "cases" / safe_case_name(case_name)
    bundle_dir = debug_root / group.id / safe_case_name(case_name)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    for name in SYSTEMC_ARTIFACT_NAMES:
        source = case_dir / name
        if source.is_file():
            shutil.copy2(source, bundle_dir / name)
    systemc_png_dir = case_dir / "systemc"
    if systemc_png_dir.is_dir():
        destination = bundle_dir / "systemc"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(systemc_png_dir, destination)

    rerun_dir = bundle_dir / "rerun"
    rerun_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.runner),
        f"--pvrgpu-output-dir={rerun_dir}",
        f"--deqp-case={case_name}",
        f"--deqp-surface-type={args.surface}",
        f"--deqp-watchdog={args.watchdog}",
        "--deqp-log-images=enable",
    ]
    if args.gl_config:
        command.append(f"--deqp-gl-config-name={args.gl_config}")
    if args.systemc_api_lib:
        command.append(f"--pvrgpu-systemc-api-lib={args.systemc_api_lib}")

    rerun_log = bundle_dir / "rerun-output.txt"
    with rerun_log.open("w", encoding="utf-8") as log_file:
        rerun = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, text=True, check=False)

    readme_lines = [
        f"# Debug bundle: {case_name}",
        "",
        f"Group: {group.id} ({group.label})",
        f"Original batch status: {case_record.get('status')} "
        f"(runner exit {case_record.get('runner_exit')}, "
        f"{case_record.get('duration_seconds')}s)",
        f"Isolated re-run exit code: {rerun.returncode}",
        "",
        "## Where to look first",
        "",
        "1. `rerun-output.txt` -- full stdout/stderr of the isolated re-run (images forced on).",
        "2. `rerun/results.qpa` -- this case's own QPA result block from the isolated re-run.",
        "3. `driver-command.txt` / `driver-counter.txt` -- what the PCO driver actually "
        "submitted to the SystemC bridge during the *original batch run* (present only if "
        "the driver got far enough to submit a command).",
        "4. `systemc.jsonl` / `systemc/*.png` -- SystemC-side simulation trace and rendered "
        "output from the *original batch run*, if the bridge ran to completion.",
        f"5. `rerun/cases/{safe_case_name(case_name)}/` -- the same three artifact kinds "
        "above, but from the isolated re-run; compare against the batch-run copies above if "
        "they look stale or the two runs disagree.",
        "",
        "## Re-run this case by hand",
        "",
        "```bash",
        " \\\n  ".join(command),
        "```",
    ]
    (bundle_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    return bundle_dir


def render_markdown_report(report: dict) -> str:
    lines = [
        "# PvrGPU Live dEQP Regression Report",
        "",
        f"Generated: {report['generated_at']}",
        f"Runner: `{report['runner']}`",
        "",
        "## Executed groups (direct dEQP -> Mesa/PCO driver -> pvrgpu SystemC bridge)",
        "",
        "| Group | Suite | Total | Pass | Skip | Warn | Fail | Duration (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in report["executed_groups"]:
        lines.append(
            f"| {group['group_id']} | {group['suite']} | {group.get('total', 0)} | "
            f"{group.get('passed', 0)} | {group.get('skipped', 0)} | "
            f"{group.get('warnings', 0)} | {group.get('failed', 0)} | "
            f"{group['duration_seconds']} |"
        )
    totals = report["totals"]
    lines += [
        "",
        f"**Overall (executed groups only): {totals['passed']} pass / {totals['skipped']} skip / "
        f"{totals['warnings']} warn / {totals['failed']} fail, out of {totals['total']} cases.**",
        "",
    ]
    if report["blocked_groups"]:
        lines += [
            "## Blocked groups (discovery-only, not executed)",
            "",
            "| Group | Suite | Catalogued cases | Live discovered | Drift | Reason |",
            "|---|---|---:|---:|---:|---|",
        ]
        for group in report["blocked_groups"]:
            lines.append(
                f"| {group['group_id']} | {group['suite']} | {group['locked_case_count']} | "
                f"{group['live_case_count']} | {group['case_count_drift']} | "
                f"{group['availability_reason']} |"
            )
        lines.append("")
    if report["failures"]:
        lines += ["## Failures", ""]
        for failure in report["failures"]:
            lines.append(
                f"- `{failure['case']}` (group `{failure['group_id']}`, status `{failure['status']}`) "
                f"-- debug bundle: `{failure['debug_bundle']}`"
            )
        lines.append("")
    else:
        lines += ["## Failures", "", "None.", ""]
    return "\n".join(lines) + "\n"


def _history_entry(source: dict, run_at: str) -> dict:
    return {
        "run_at": run_at,
        "output_dir": source.get("output_dir"),
        "list_only": bool(source.get("list_only")),
        "total": source.get("total", 0),
        "passed": source.get("passed", 0),
        "skipped": source.get("skipped", 0),
        "warnings": source.get("warnings", 0),
        "failed": source.get("failed", 0),
        "duration_seconds": source.get("duration_seconds", 0),
        "live_case_count": source.get("live_case_count", 0),
        "case_count_drift": source.get("case_count_drift", 0),
    }


def load_dashboard_state(state_path: Path) -> dict:
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
    else:
        state = {}
    state.setdefault("schema", DASHBOARD_SCHEMA)
    state.setdefault("groups", {})
    state.setdefault("runs", [])
    return state


def update_dashboard_state(state: dict, report: dict) -> dict:
    """Merge one run's report into the cumulative dashboard state.

    Every group in the 24-entry catalog gets a row (seeded here on first
    sight) so the dashboard always shows the full catalog, not just whatever
    subset a given run happened to select; only groups this run actually
    touched get a new history entry appended.
    """
    run_at = report["generated_at"]
    groups_state: dict = state.setdefault("groups", {})

    # Seed/refresh static catalog metadata for all 24 groups every run, so
    # availability flips (e.g. a group unblocked later) show up immediately
    # even for groups this run didn't touch.
    for group in GROUP_SPECS:
        row = groups_state.setdefault(
            group.id,
            {
                "label": group.label,
                "suite": group.suite,
                "last_run_at": None,
                "last_result": None,
                "history": [],
            },
        )
        row["label"] = group.label
        row["suite"] = group.suite
        row["available"] = group.available
        row["availability_reason"] = group.availability_reason
        row["locked_case_count"] = group.locked_case_count

    for source in (*report["executed_groups"], *report["blocked_groups"]):
        row = groups_state[source["group_id"]]
        entry = _history_entry(source, run_at)
        row["last_run_at"] = run_at
        row["last_result"] = entry
        history = row.setdefault("history", [])
        history.append(entry)
        del history[:-MAX_HISTORY_PER_GROUP]

    run_log_entry = {
        "run_at": run_at,
        "output_dir": report.get("output_dir"),
        "runner": report["runner"],
        "groups_requested": report["groups_requested"],
        "totals": report["totals"],
        "executed_group_ids": [g["group_id"] for g in report["executed_groups"]],
        "blocked_group_ids": [g["group_id"] for g in report["blocked_groups"]],
        "failed_case_count": len(report["failures"]),
    }
    runs_log: list = state.setdefault("runs", [])
    runs_log.append(run_log_entry)
    del runs_log[:-MAX_RUN_LOG]
    return state


def _relative_link(target: str | None, dashboard_dir: Path) -> str | None:
    if not target:
        return None
    try:
        return os.path.relpath(target, start=dashboard_dir)
    except ValueError:
        return target


def _fail_trend(history: list[dict]) -> str:
    recent = history[-8:]
    parts = []
    for entry in recent:
        if entry.get("list_only"):
            drift = entry.get("case_count_drift", 0)
            parts.append("0" if drift == 0 else f"{drift:+d}")
        else:
            parts.append(str(entry.get("failed", 0)))
    return " ".join(parts) if parts else "—"


def render_dashboard_html(state: dict, dashboard_dir: Path) -> str:
    runs_log = state.get("runs", [])
    last_run_at = runs_log[-1]["run_at"] if runs_log else None
    groups_state = state.get("groups", {})

    # Preserve the catalog's own grouping/order (EGL, then GLES3 functional,
    # GLES3 stress, GLES31, GLES32) rather than inventing new categories.
    suite_sections: dict[str, list] = {}
    for group in GROUP_SPECS:
        suite_sections.setdefault(group.suite, []).append(group)

    def esc(value: object) -> str:
        text_value = "" if value is None else str(value)
        return (
            text_value.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    sections_html = []
    for suite, groups in suite_sections.items():
        rows = []
        for group in groups:
            row = groups_state.get(group.id, {})
            last_run = row.get("last_run_at")
            last_result = row.get("last_result") or {}
            history = row.get("history", [])
            available = row.get("available", group.available)
            reason = row.get("availability_reason", group.availability_reason)

            status_html = (
                '<span class="badge badge-ok">available</span>'
                if available
                else f'<span class="badge badge-blocked" title="{esc(reason)}">blocked</span>'
            )

            if not last_run:
                result_html = '<span class="muted">never run</span>'
            elif last_result.get("list_only"):
                drift = last_result.get("case_count_drift", 0)
                drift_html = (
                    f'<span class="drift-zero">0</span>'
                    if drift == 0
                    else f'<span class="drift-nonzero">{drift:+d}</span>'
                )
                result_html = (
                    f'discovery: {last_result.get("live_case_count", 0)} cases '
                    f'(catalog {group.locked_case_count}, drift {drift_html})'
                )
            else:
                result_html = (
                    f'<span class="pass">{last_result.get("passed", 0)}P</span> '
                    f'<span class="skip">{last_result.get("skipped", 0)}S</span> '
                    f'<span class="warn">{last_result.get("warnings", 0)}W</span> '
                    f'<span class="fail">{last_result.get("failed", 0)}F</span> '
                    f'/ {last_result.get("total", 0)}'
                )

            link_target = _relative_link(last_result.get("output_dir"), dashboard_dir)
            link_html = (
                f'<a href="{esc(link_target)}/report.md" class="run-link">view run output</a>'
                if link_target
                else '<span class="muted">\u2014</span>'
            )

            rows.append(
                "<tr>"
                f'<td class="group-id">{esc(group.id)}</td>'
                f"<td>{esc(group.label)}</td>"
                f"<td>{status_html}</td>"
                f'<td class="mono">{esc(last_run) or "—"}</td>'
                f"<td>{result_html}</td>"
                f'<td class="mono trend" title="most recent {len(history)} run(s), oldest to newest">'
                f"{esc(_fail_trend(history))}</td>"
                f'<td>{len(history)} run(s)</td>'
                f"<td>{link_html}</td>"
                "</tr>"
            )
        sections_html.append(
            f'<h2>{esc(suite)}</h2>\n<table><thead><tr>'
            "<th>Group</th><th>Label</th><th>Status</th><th>Last run (UTC)</th>"
            "<th>Latest result</th><th>Fail/drift trend</th><th>Runs logged</th><th>Link</th>"
            "</tr></thead><tbody>\n" + "\n".join(rows) + "\n</tbody></table>"
        )

    recent_runs_rows = []
    for entry in reversed(runs_log[-15:]):
        totals = entry.get("totals", {})
        recent_runs_rows.append(
            "<tr>"
            f'<td class="mono">{esc(entry.get("run_at"))}</td>'
            f'<td>{esc(entry.get("groups_requested"))}</td>'
            f'<td>{len(entry.get("executed_group_ids", []))}</td>'
            f'<td>{len(entry.get("blocked_group_ids", []))}</td>'
            f'<td><span class="pass">{totals.get("passed", 0)}P</span> '
            f'<span class="skip">{totals.get("skipped", 0)}S</span> '
            f'<span class="warn">{totals.get("warnings", 0)}W</span> '
            f'<span class="fail">{totals.get("failed", 0)}F</span></td>'
            f'<td>{entry.get("failed_case_count", 0)}</td>'
            "</tr>"
        )

    generated_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PvrGPU Live dEQP Dashboard</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", sans-serif; margin: 2rem; color: #1a1a1a; background: #fafafa; }}
  h1 {{ margin-bottom: 0.2rem; }}
  .meta {{ color: #555; margin-bottom: 1.5rem; }}
  h2 {{ margin-top: 2.2rem; border-bottom: 2px solid #ddd; padding-bottom: 0.3rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1rem; background: #fff; }}
  th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #e5e5e5; font-size: 0.9rem; }}
  th {{ background: #f0e9e1; }}
  .group-id {{ font-family: ui-monospace, monospace; font-size: 0.82rem; }}
  .mono {{ font-family: ui-monospace, monospace; font-size: 0.82rem; }}
  .muted {{ color: #999; }}
  .badge {{ padding: 2px 8px; border-radius: 10px; font-size: 0.78rem; }}
  .badge-ok {{ background: #e2f3e6; color: #1e6b34; }}
  .badge-blocked {{ background: #f3e6e0; color: #8a3b12; cursor: help; }}
  .pass {{ color: #1e6b34; }}
  .skip {{ color: #7a6a00; }}
  .warn {{ color: #a15c00; }}
  .fail {{ color: #a11212; font-weight: 600; }}
  .drift-zero {{ color: #1e6b34; }}
  .drift-nonzero {{ color: #a11212; font-weight: 600; }}
  .run-link {{ color: #8a3b12; }}
</style>
</head>
<body>
<h1>PvrGPU Live dEQP Dashboard</h1>
<p class="meta">
  Cumulative across every <code>run_deqp_regression.sh</code> invocation &mdash; direct dEQP &rarr; Mesa/PCO driver &rarr; pvrgpu SystemC bridge, no RDC capture involved.<br>
  Last regression run: <span class="mono">{esc(last_run_at) or "never"}</span> &middot;
  Dashboard rendered: <span class="mono">{esc(generated_at)}</span> &middot;
  {len(runs_log)} run(s) logged.
</p>
{"".join(sections_html)}
<h2>Recent runs</h2>
<table><thead><tr>
<th>Run at (UTC)</th><th>--groups</th><th>Executed</th><th>Blocked/listed</th><th>Totals</th><th>Failing cases</th>
</tr></thead><tbody>
{"".join(recent_runs_rows) if recent_runs_rows else '<tr><td colspan="6" class="muted">No runs logged yet.</td></tr>'}
</tbody></table>
</body>
</html>
"""


def write_dashboard(dashboard_dir: Path, report: dict) -> Path:
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    state_path = dashboard_dir / "state.json"
    html_path = dashboard_dir / "dashboard.html"

    state = load_dashboard_state(state_path)
    state = update_dashboard_state(state, report)

    state_tmp = state_path.with_suffix(".json.tmp")
    state_tmp.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(state_tmp, state_path)

    html = render_dashboard_html(state, dashboard_dir)
    html_tmp = html_path.with_suffix(".html.tmp")
    html_tmp.write_text(html, encoding="utf-8")
    os.replace(html_tmp, html_path)
    return html_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    args.runner = args.runner.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    if not args.runner.is_file():
        print(f"deqp regression: pvrgpu-deqp not found: {args.runner}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups_dir = args.output_dir / "groups"
    debug_dir = args.output_dir / "debug"

    try:
        selected = select_groups(args.groups)
    except KeyError as error:
        print(f"deqp regression: {error}", file=sys.stderr)
        return 2
    if not selected:
        print("deqp regression: no groups matched --groups", file=sys.stderr)
        return 2

    executed_groups: list[dict] = []
    blocked_groups: list[dict] = []
    failures: list[dict] = []

    for group in selected:
        group_output_dir = groups_dir / group.id
        if not group.available:
            if not args.include_blocked:
                continue
            blocked_groups.append(run_group(args, group, group_output_dir, list_only=True))
            continue

        result = run_group(args, group, group_output_dir, list_only=False)
        executed_groups.append(result)
        for case_record in load_failed_cases(group_output_dir):
            debug_bundle = None
            if not args.no_debug_bundle:
                bundle_path = build_debug_bundle(args, group, group_output_dir, case_record, debug_dir)
                debug_bundle = str(bundle_path.relative_to(args.output_dir))
            failures.append(
                {
                    "group_id": group.id,
                    "case": case_record["case"],
                    "status": case_record.get("status"),
                    "runner_exit": case_record.get("runner_exit"),
                    "debug_bundle": debug_bundle,
                }
            )

    totals = {
        "total": sum(group.get("total", 0) for group in executed_groups),
        "passed": sum(group.get("passed", 0) for group in executed_groups),
        "skipped": sum(group.get("skipped", 0) for group in executed_groups),
        "warnings": sum(group.get("warnings", 0) for group in executed_groups),
        "failed": sum(group.get("failed", 0) for group in executed_groups),
    }
    report = {
        "schema": SCHEMA,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "runner": str(args.runner),
        "output_dir": str(args.output_dir),
        "groups_requested": args.groups,
        "executed_groups": executed_groups,
        "blocked_groups": blocked_groups,
        "totals": totals,
        "failures": failures,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(render_markdown_report(report), encoding="utf-8")

    print()
    print(f"Report written to {args.output_dir / 'report.md'} and report.json")
    print(
        f"Totals: {totals['passed']} pass / {totals['skipped']} skip / {totals['warnings']} warn / "
        f"{totals['failed']} fail (of {totals['total']} cases across {len(executed_groups)} groups)"
    )
    if failures:
        print(f"{len(failures)} failing case(s); see debug/ bundles under {args.output_dir}")

    if not args.no_dashboard:
        dashboard_dir = args.dashboard_dir or (args.output_dir.parent / "dashboard")
        dashboard_dir = dashboard_dir.expanduser().resolve()
        dashboard_path = write_dashboard(dashboard_dir, report)
        print(f"Dashboard updated: {dashboard_path}")

    exit_code = 0
    if totals["failed"] > 0:
        exit_code = 1
    if args.fail_on_blocked_drift and any(group["case_count_drift"] != 0 for group in blocked_groups):
        exit_code = exit_code or 3
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
