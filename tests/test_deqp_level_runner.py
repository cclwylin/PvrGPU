"""Behavior checks for the headless level runner with an isolated fake runtime."""

import csv
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from script.deqp_4level_catalog import (
    DEFAULT_GL_CONFIG,
    GROUPS_4LEVEL,
    MS4_GL_CONFIG,
    TIERS_BY_ID,
    build_plan,
    is_long_only_case,
)

RUNNER = REPO / "script/run_deqp_level.sh"
FIELDS = ["case", "status", "exit_code", "duration_ms", "case_dir"]

FAKE_DRIVER = r'''
import csv
import json
import os
from pathlib import Path
import sys
import time

args = sys.argv[1:]
def value(flag):
    return args[args.index(flag) + 1]

config = json.loads(Path(os.environ['DEQP_LEVEL_TEST_CONFIG']).read_text())
kind = 'check' if '--check' in args else 'discover' if '--discover' in args else 'run'
module = value('--module')
def event(phase):
    payload = json.dumps(dict(kind=kind, phase=phase, module=module, pid=os.getpid(),
                              argv=args, time=time.monotonic())) + '\n'
    fd = os.open(os.environ['DEQP_LEVEL_TEST_EVENTS'],
                 os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, payload.encode())
    finally:
        os.close(fd)

event('start')
if kind == 'check':
    event('end')
    sys.exit(config.get('preflight_exit', 0))
if kind == 'discover':
    source = Path(os.environ['DEQP_LEVEL_TEST_DISCOVERY']) / (module + '.txt')
    Path(value('--caselist-out')).write_text(source.read_text())
    event('end')
    sys.exit(0)

time.sleep(config.get('delay', 0))
output = Path(value('--output-dir'))
output.mkdir(parents=True)
cases = Path(value('--caselist')).read_text().splitlines()
if not config.get('missing_summary'):
    with (output / 'summary.tsv').open('w', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(['case', 'status', 'exit_code', 'duration_ms', 'case_dir'])
        for case in cases:
            if case in config.get('omit', []):
                continue
            case_options = config.get('cases', {}).get(case, {})
            row = [case, case_options.get('status', 'Pass'),
                   case_options.get('exit_code', '0'),
                   case_options.get('duration_ms', '1'), str(output / case)]
            if case_options.get('extra_column'):
                row.append('unexpected')
            writer.writerow(row)
            if case_options.get('duplicate'):
                writer.writerow(row)
event('end')
sys.exit(config.get('job_exit_by_module', {}).get(module, 0))
'''


class DeqpLevelRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="deqp level runner ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.discovery_dir = self.root / "discovery cache"
        self.discovery_dir.mkdir()
        self.discovery = {module: [] for module in ("egl", "gles2", "gles3", "gles31")}
        for group in GROUPS_4LEVEL:
            self.discovery[group.module].append(group.selectors[0] + "fixture")
        self.discovery["gles2"].append("dEQP-GLES2.functional.fixture")
        self.discovery["gles3"].append("dEQP-GLES3.functional.outside_catalog.fixture")
        self.ms4_cases = [
            "dEQP-GLES3.functional.multisample.default_framebuffer.depth",
            "dEQP-GLES31.functional.multisample.default_framebuffer.depth",
        ]
        self.discovery["gles3"].append(self.ms4_cases[0])
        self.discovery["gles31"].append(self.ms4_cases[1])
        self.discovery["gles3"].append(
            "dEQP-GLES3.functional.multisample.fbo_max_samples.constancy_alpha_to_coverage"
        )
        # Interleave enough short/long image cases to exercise the L1 quota.
        for index in range(125):
            self.discovery["gles31"].append(
                f"dEQP-GLES31.functional.image_load_store.2d.fixture_{index:03d}"
            )
            if index % 5 == 0:
                self.discovery["gles31"].append(
                    f"dEQP-GLES31.functional.image_load_store.3d.atomic.fixture_{index:03d}"
                )
        for module, cases in self.discovery.items():
            (self.discovery_dir / f"{module}.txt").write_text("\n".join(cases) + "\n")
        self.events = self.root / "runner events.jsonl"
        self.config_path = self.root / "fake configuration.json"
        self.config_path.write_text("{}")
        fake_driver = self.root / "fake dynamic runner.py"
        fake_driver.write_text(FAKE_DRIVER)
        self.fake_runner = self.root / "fake dynamic runner.sh"
        self.fake_runner.write_text(
            "#!/usr/bin/env bash\nexec " + shlex.quote(sys.executable) + " "
            + shlex.quote(str(fake_driver)) + ' "$@"\n'
        )
        self.env = dict(os.environ,
                        PVRGPU_DEQP_LEVEL_PYTHON=sys.executable,
                        PVRGPU_OUTPUT_ROOT=str(self.root / "environment output"),
                        DEQP_LEVEL_TEST_CONFIG=str(self.config_path),
                        DEQP_LEVEL_TEST_EVENTS=str(self.events),
                        DEQP_LEVEL_TEST_DISCOVERY=str(self.discovery_dir))
        self.counter = 0

    def invoke(self, *args, config=None, output=None, discovery_dir=None):
        self.counter += 1
        if config is not None:
            self.config_path.write_text(json.dumps(config))
        output = output or self.root / f"run output {self.counter}"
        command = ["bash", str(RUNNER), "--discovery-dir",
                   str(discovery_dir or self.discovery_dir), "--runner", str(self.fake_runner),
                   "--output-dir", str(output), *map(str, args)]
        completed = subprocess.run(command, env=self.env, capture_output=True,
                                   text=True, timeout=30)
        return completed, output

    def read_plan(self, output):
        return json.loads((output / "plan.json").read_text())

    def read_result(self, output):
        self.assertTrue((output / "result.json").is_file(), "runner must preserve a result report")
        return json.loads((output / "result.json").read_text())

    def event_rows(self):
        if not self.events.is_file():
            return []
        return [json.loads(line) for line in self.events.read_text().splitlines()]

    def expected_plan(self, tier_id):
        tier = TIERS_BY_ID[tier_id]
        return build_plan(name=tier_id, tier=tier, groups=list(GROUPS_4LEVEL),
                          discovery=self.discovery, shards=tier.shards,
                          log_images=tier.log_images, default_config=DEFAULT_GL_CONFIG,
                          whole_modules=tier.whole_modules)

    def test_default_and_all_tiers_use_shared_plan_without_launching_runtime(self):
        for flag, tier_id in ((None, "L1"), ("--1", "L1"), ("--2", "L2"),
                              ("--3", "L3"), ("--4", "L4")):
            with self.subTest(flag=flag):
                args = ["--dry-run"] + ([flag] if flag else [])
                completed, output = self.invoke(*args)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                plan = self.read_plan(output)
                expected = self.expected_plan(tier_id)
                cases = (output / "caselists/all.txt").read_text().splitlines()
                self.assertEqual(plan["tier"], tier_id)
                self.assertEqual(cases, [case for bucket in expected.buckets for case in bucket.cases])
                self.assertEqual(plan["total_cases"], expected.total_cases)
                self.assertEqual(plan["long_excluded_l4_only"], expected.long_excluded_count)
                selected_jobs = [Path(job["caselist"]).read_text().splitlines() for job in plan["jobs"]]
                self.assertCountEqual([case for cases_in_job in selected_jobs for case in cases_in_job], cases)
                for case in self.ms4_cases:
                    job = next(job for job in plan["jobs"] if case in Path(job["caselist"]).read_text().splitlines())
                    self.assertEqual(job["gl_config"], MS4_GL_CONFIG)
                if tier_id == "L4":
                    self.assertEqual({job["module"] for job in plan["jobs"]}, set(self.discovery))
                    self.assertEqual(plan["shards"], 1)
                    self.assertEqual(plan["log_images"], "enable")
                    self.assertTrue(any(is_long_only_case(case) for case in cases))
                else:
                    self.assertFalse(any(is_long_only_case(case) for case in cases))
        self.assertEqual(self.event_rows(), [])

    def test_l4_runs_all_modules_serially_and_enables_images(self):
        completed, output = self.invoke("--4", config={"delay": 0.02})
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = self.read_result(output)
        self.assertEqual(result["completed"], sum(map(len, self.discovery.values())))
        self.assertEqual(result["statuses"], {"Pass": result["total"]})
        active = 0
        modules = set()
        for event in self.event_rows():
            if event["kind"] != "run":
                continue
            active += 1 if event["phase"] == "start" else -1
            self.assertIn(active, (0, 1))
            modules.add(event["module"])
            args = event["argv"]
            self.assertEqual(args[args.index("--log-images") + 1], "enable")
        self.assertEqual(active, 0)
        self.assertEqual(modules, set(self.discovery))

    def test_merge_preserves_raw_statuses_and_flags_fail_or_nonzero_case_exit(self):
        cases = [case for bucket in self.expected_plan("L1").buckets for case in bucket.cases]
        statuses = ("NotSupported", "CompatibilityWarning", "QualityWarning", "Waiver", "Fail")
        options = {case: {"status": status} for case, status in zip(cases, statuses)}
        options[cases[5]] = {"status": "Pass", "exit_code": 17}
        completed, output = self.invoke("--shards", "3", "--timeout", "9", "--verify-link",
                                        config={"cases": options})
        self.assertEqual(completed.returncode, 1, completed.stderr)
        result = self.read_result(output)
        self.assertEqual(result["completed"], len(cases))
        self.assertEqual({row["case"] for row in result["failures"]}, {cases[4], cases[5]})
        self.assertEqual(result["job_failures"], {})
        self.assertEqual(result["incomplete_cases"], [])
        self.assertEqual(result["evidence_issues"], [])
        for status in statuses:
            self.assertEqual(result["statuses"][status], 1)
        with (output / "summary.tsv").open(newline="") as stream:
            merged = list(csv.DictReader(stream, delimiter="\t"))
        self.assertEqual([row["case"] for row in merged], cases)
        for job in self.read_plan(output)["jobs"]:
            args = job["command"]
            self.assertIn("--verify-link", args)
            self.assertEqual(args[args.index("--timeout") + 1], "9")

    def test_nonfailure_statuses_return_success(self):
        cases = [case for bucket in self.expected_plan("L1").buckets for case in bucket.cases]
        options = {case: {"status": status} for case, status in zip(cases,
                   ("NotSupported", "CompatibilityWarning", "QualityWarning", "Waiver"))}
        completed, output = self.invoke(config={"cases": options})
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(self.read_result(output)["failures"], [])

    def test_missing_case_and_runner_job_failure_are_reported(self):
        case = self.discovery["egl"][0]
        completed, output = self.invoke("--shards", "1", config={
            "omit": [case], "job_exit_by_module": {"gles31": 17}})
        self.assertEqual(completed.returncode, 1, completed.stderr)
        result = self.read_result(output)
        self.assertEqual(result["incomplete_cases"], [case])
        self.assertTrue(result["job_failures"])
        self.assertEqual(set(result["job_failures"].values()), {17})
        self.assertEqual(result["failures"], [])

    def test_missing_summary_is_not_reported_as_success(self):
        completed, output = self.invoke("--shards", "1", config={"missing_summary": True})
        self.assertEqual(completed.returncode, 1, completed.stderr)
        result = self.read_result(output)
        self.assertEqual(result["completed"], 0)
        self.assertEqual(len(result["incomplete_cases"]), result["total"])

    def test_malformed_summary_rows_keep_a_result_report(self):
        case = self.discovery["egl"][0]
        for option in ({"duration_ms": "invalid"}, {"duration_ms": "-1"},
                       {"exit_code": "invalid"}, {"duplicate": True}, {"extra_column": True}):
            with self.subTest(option=option):
                completed, output = self.invoke("--shards", "1", config={"cases": {case: option}})
                self.assertEqual(completed.returncode, 1, completed.stderr)
                result = self.read_result(output)
                self.assertTrue(result["evidence_issues"])
                if not option.get("duplicate"):
                    self.assertIn(case, result["incomplete_cases"])

    def test_preflight_failure_stops_before_test_jobs(self):
        completed, _ = self.invoke(config={"preflight_exit": 19})
        self.assertEqual(completed.returncode, 1, completed.stderr)
        self.assertIn("19", completed.stderr)
        self.assertEqual({event["kind"] for event in self.event_rows()}, {"check"})

    def test_sigterm_cancels_run_and_stops_its_children(self):
        self.config_path.write_text(json.dumps({"delay": 30}))
        output = self.root / "cancelled output"
        command = ["bash", str(RUNNER), "--shards", "2", "--runner", str(self.fake_runner),
                   "--discovery-dir", str(self.discovery_dir), "--output-dir", str(output)]
        process = subprocess.Popen(command, env=self.env, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)

        def running_children():
            active = []
            pids = {event["pid"] for event in self.event_rows() if event["kind"] == "run"}
            for pid in pids:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    continue
                active.append(pid)
            return active

        try:
            deadline = time.monotonic() + 10
            while not running_children() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.025)
            self.assertTrue(running_children(), "fake runtime did not start within 10 seconds")
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=8)
            self.assertEqual(process.returncode, 130, stdout + stderr)
            result = self.read_result(output)
            self.assertTrue(result["cancelled"])
            self.assertTrue(result["incomplete_cases"])
            deadline = time.monotonic() + 2
            while running_children() and time.monotonic() < deadline:
                time.sleep(0.025)
            self.assertEqual(running_children(), [], "cancelled runtime children remain alive")
        finally:
            # Bound failures too, so a cancellation regression cannot leak test processes.
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=3)
            for pid in running_children():
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_refresh_discovery_validates_and_publishes_cache(self):
        cache = self.root / "new cache"
        completed, output = self.invoke("--4", "--refresh-discovery", discovery_dir=cache)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for module, cases in self.discovery.items():
            self.assertEqual((cache / f"{module}.txt").read_text().splitlines(), cases)
        self.assertEqual(self.read_result(output)["completed"], sum(map(len, self.discovery.values())))
        discovered = {event["module"] for event in self.event_rows() if event["kind"] == "discover"}
        self.assertEqual(discovered, set(self.discovery))

    def test_output_directory_cannot_be_overwritten(self):
        output = self.root / "existing results"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("existing evidence")
        completed, _ = self.invoke("--dry-run", output=output)
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(marker.read_text(), "existing evidence")
        self.assertEqual(list(output.iterdir()), [marker])
        self.assertEqual(self.event_rows(), [])

    def test_auto_output_and_runtime_flags_preserve_environment_precedence(self):
        env = dict(self.env, PVRGPU_MESA_PVRGPU_PREFIX=str(self.root / "environment mesa"))
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                output_root = self.root / ("explicit output" if explicit else "environment output")
                mesa_prefix = self.root / ("explicit mesa" if explicit else "environment mesa")
                command = ["bash", str(RUNNER), "--dry-run", "--runner", str(self.fake_runner),
                           "--discovery-dir", str(self.discovery_dir)]
                if explicit:
                    command += ["--output-root", str(output_root), "--mesa-prefix", str(mesa_prefix)]
                completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                generated = list((output_root / "deqp_4level").iterdir())
                self.assertEqual(len(generated), 1)
                for job in self.read_plan(generated[0])["jobs"]:
                    args = job["command"]
                    self.assertEqual(args[args.index("--mesa-prefix") + 1], str(mesa_prefix))
        self.assertEqual(self.event_rows(), [])

    def test_help_needs_no_runtime(self):
        completed = subprocess.run(["bash", str(RUNNER), "--help"], env=self.env,
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for flag in ("--1", "--2", "--3", "--4", "--dry-run"):
            self.assertIn(flag, completed.stdout)
        self.assertEqual(self.event_rows(), [])

    def test_invalid_arguments_fail_without_launching_runtime(self):
        for args in (("--1", "--2"), ("--0",), ("--5",), ("--shards", "0"),
                     ("--shards", "9"), ("--4", "--shards", "2"),
                     ("--timeout", "-1"), ("--dry-run", "--refresh-discovery")):
            with self.subTest(args=args):
                completed, output = self.invoke(*args)
                self.assertEqual(completed.returncode, 2, completed.stderr)
                self.assertFalse(output.exists())
        self.assertEqual(self.event_rows(), [])

    def test_dry_run_rejects_missing_invalid_or_duplicate_cache(self):
        module_path = self.discovery_dir / "egl.txt"
        for content in ("", "dEQP-GLES3.functional.wrong_module\n",
                        "dEQP-EGL.functional.bad*\n", "dEQP-EGL.a\ndEQP-EGL.a\n"):
            with self.subTest(content=content):
                module_path.write_text(content)
                completed, _ = self.invoke("--dry-run")
                self.assertEqual(completed.returncode, 1)
        module_path.unlink()
        completed, _ = self.invoke("--dry-run")
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(self.event_rows(), [])


if __name__ == "__main__":
    unittest.main()
