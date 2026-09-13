from pathlib import Path
import os
import subprocess
import tempfile
import unittest


RUNNER = Path(__file__).resolve().parents[1] / "script/run_deqp_dynamic.sh"


class DynamicRunnerTests(unittest.TestCase):
    def test_process_exit_survives_optional_loader_filter(self) -> None:
        source = RUNNER.read_text()
        start = source.index('    case_start_ms="$(now_ms)"')
        end = source.index('    case_end_ms="$(now_ms)"', start)
        # Execute the actual runner pipeline, not a second implementation.
        pipeline = source[start:end]
        with tempfile.TemporaryDirectory(prefix="pvrgpu-runner-test-") as output:
            for verify in (0, 1):
                for code in (0, 1, 17, 124, 137):
                    for loader_only in (0, 1):
                        with self.subTest(verify=verify, code=code, loader_only=loader_only):
                            shell = r'''
set -uo pipefail
now_ms() { printf '0'; }
run_one_case() {
    printf 'dyld[123]: loaded library\n'
    if (( ! loader_only )); then printf 'test output\n'; fi
    return "${test_exit}"
}
''' + pipeline + '\nprintf "captured_exit=%s\\n" "${exit_code}"\n'
                            env = dict(os.environ, opt_verify_link=str(verify),
                                       test_exit=str(code), loader_only=str(loader_only),
                                       log_path=str(Path(output) / "run.log"))
                            result = subprocess.run(["bash", "-c", shell], env=env,
                                                    capture_output=True, text=True, check=True)
                            self.assertIn(f"captured_exit={code}\n", result.stdout)
                            self.assertIn("dyld[123]", (Path(output) / "run.log").read_text())

    def test_default_memory_mode_matches_graphics_and_preserves_override(self) -> None:
        line = next(line for line in RUNNER.read_text().splitlines()
                    if line.startswith("export PVRGPU_MODEL_MEMORY_MODE="))
        for value, expected in ((None, "cache"), ("", "cache"), ("cache", "cache"),
                                ("direct", "direct"), ("bypass", "bypass")):
            with self.subTest(value=value):
                env = dict(os.environ)
                env.pop("PVRGPU_MODEL_MEMORY_MODE", None)
                if value is not None:
                    env["PVRGPU_MODEL_MEMORY_MODE"] = value
                result = subprocess.run(["bash", "-c", line +
                                         '\nprintf "%s" "$PVRGPU_MODEL_MEMORY_MODE"'],
                                        env=env, capture_output=True, text=True, check=True)
                self.assertEqual(result.stdout, expected)

    def test_default_texture_memory_path_is_short_and_preserves_override(self) -> None:
        line = next(line for line in RUNNER.read_text().splitlines()
                    if line.startswith("export PVRGPU_TEXTURE_MEMORY_PATH="))
        for value, expected in ((None, "short"), ("", "short"),
                                ("short", "short"), ("cached", "cached")):
            with self.subTest(value=value):
                env = dict(os.environ)
                env.pop("PVRGPU_TEXTURE_MEMORY_PATH", None)
                if value is not None:
                    env["PVRGPU_TEXTURE_MEMORY_PATH"] = value
                result = subprocess.run(
                    ["bash", "-c", line +
                     '\nprintf "%s" "$PVRGPU_TEXTURE_MEMORY_PATH"'],
                    env=env, capture_output=True, text=True, check=True)
                self.assertEqual(result.stdout, expected)

    def test_texture_memory_path_cli_rejects_empty_and_invalid_values(self) -> None:
        for value in ("", "invalid"):
            with self.subTest(value=value):
                result = subprocess.run(
                    [RUNNER, "--texture-memory-path", value],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "texture memory path must be short or cached",
                    result.stderr,
                )

    def test_texture_memory_path_environment_wins_over_local_env(self) -> None:
        source = RUNNER.read_text()
        start = source.index('if [[ -f "${REPO_DIR}/config/local.env" ]]')
        end = source.index('\nPVRGPU_BUILD_DIR=', start)
        config_block = source[start:end]
        with tempfile.TemporaryDirectory(prefix="pvrgpu-runner-config-") as root:
            config = Path(root) / "config"
            config.mkdir()
            (config / "local.env").write_text(
                "PVRGPU_TEXTURE_MEMORY_PATH=short\n"
                "PVRGPU_MODEL_MEMORY_MODE=cache\n"
                "PVRGPU_TEXTURE_LOD_MODE=exact\n"
            )
            shell = (
                'opt_mesa_prefix=""; opt_systemc_lib=""\n'
                + config_block
                + '\nprintf "%s|%s|%s" "$PVRGPU_TEXTURE_MEMORY_PATH" '
                  '"$PVRGPU_MODEL_MEMORY_MODE" "$PVRGPU_TEXTURE_LOD_MODE"\n'
            )
            env = dict(
                os.environ,
                REPO_DIR=root,
                PVRGPU_TEXTURE_MEMORY_PATH="cached",
                PVRGPU_MODEL_MEMORY_MODE="cache",
                PVRGPU_TEXTURE_LOD_MODE="llvmpipe",
            )
            result = subprocess.run(
                ["bash", "-c", shell], env=env, capture_output=True,
                text=True, check=True
            )
            self.assertEqual(result.stdout, "cached|cache|llvmpipe")

    def test_default_texture_lod_mode_is_exact_and_preserves_override(self) -> None:
        line = next(line for line in RUNNER.read_text().splitlines()
                    if line.startswith("export PVRGPU_TEXTURE_LOD_MODE="))
        for value, expected in ((None, "exact"), ("", "exact"),
                                ("exact", "exact"),
                                ("llvmpipe", "llvmpipe")):
            with self.subTest(value=value):
                env = dict(os.environ)
                env.pop("PVRGPU_TEXTURE_LOD_MODE", None)
                if value is not None:
                    env["PVRGPU_TEXTURE_LOD_MODE"] = value
                result = subprocess.run(
                    ["bash", "-c", line +
                     '\nprintf "%s" "$PVRGPU_TEXTURE_LOD_MODE"'],
                    env=env, capture_output=True, text=True, check=True)
                self.assertEqual(result.stdout, expected)


if __name__ == "__main__":
    unittest.main()
