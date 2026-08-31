from __future__ import annotations

from pathlib import Path
import re
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CMAKE_FILE = PROJECT_ROOT / "CMakeLists.txt"
CONFIG_EXAMPLE = PROJECT_ROOT / "config" / "local.env.example"
RUNNER_ROOT = PROJECT_ROOT / "src" / "rdc_runner"
RUNNER_MAIN = RUNNER_ROOT / "main.cpp"
RUNTIME_CONFIG = RUNNER_ROOT / "runtime_config.cpp"
PROCESS_RUNNER = RUNNER_ROOT / "process.cpp"
DRIVER_ROOT = PROJECT_ROOT / "src" / "gallium" / "drivers" / "pvrgpu"
COUNTER_MESA_ZERO_OUTPUT_PATCH = (
    PROJECT_ROOT / "third_party" / "mesa-26.2.1-llvmpipe-gs-zero-output-stats.patch"
)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class NativeMesaIntegrationTests(unittest.TestCase):
    def test_cmake_builds_and_installs_both_native_rdc_players(self) -> None:
        cmake = read_text(CMAKE_FILE)

        self.assertIn(
            'set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")', cmake
        )
        self.assertIn("add_library(pvrgpu-rdc-runner-support STATIC", cmake)
        for source in (
            "src/rdc_runner/process.cpp",
            "src/rdc_runner/runtime_config.cpp",
            "src/rdc_runner/sha256.cpp",
        ):
            self.assertIn(source, cmake)
        self.assertIn('PVRGPU_SOURCE_DIR="${CMAKE_SOURCE_DIR}"', cmake)

        for target, backend in (("llvmpipe", "llvmpipe"), ("pvrgpu", "pvrgpu")):
            with self.subTest(target=target):
                self.assertIn(
                    f"add_executable({target} src/rdc_runner/main.cpp)", cmake
                )
                self.assertIn(
                    f'target_compile_definitions({target} PRIVATE '
                    f'PVRGPU_RDC_BACKEND="{backend}")',
                    cmake,
                )
                self.assertIn(
                    f"target_link_libraries({target} PRIVATE "
                    "pvrgpu-rdc-runner-support)",
                    cmake,
                )

        self.assertIn(
            "install(TARGETS llvmpipe pvrgpu RUNTIME DESTINATION bin)", cmake
        )

    def test_native_process_and_config_layers_replace_shell_bootstrap(self) -> None:
        runtime = read_text(RUNTIME_CONFIG)
        process = read_text(PROCESS_RUNNER)

        self.assertIn('project_root_ / "config" / "local.env"', runtime)
        self.assertIn('values_["PVRGPU_PROJECT_ROOT"]', runtime)
        self.assertIn('values_["PVRGPU_WORK_ROOT"]', runtime)
        self.assertIn("value.find('`')", runtime)
        self.assertIn('value.find("$(")', runtime)
        self.assertIn("ResolveExecutable", runtime)
        self.assertIn('with_extension += ".exe"', runtime)

        self.assertIn("CreateProcessW", process)
        self.assertIn("CREATE_NEW_PROCESS_GROUP", process)
        self.assertIn("fork()", process)
        self.assertIn("execv(request.executable.c_str()", process)
        self.assertIn("unsetenv(name.c_str())", process)
        self.assertIn("setenv(name.c_str(), value.c_str(), 1)", process)
        self.assertNotIn("std::system(", process)
        self.assertNotIn("popen(", process)

    def test_native_cli_binds_one_rdc_to_result_and_artifact_schema(self) -> None:
        runner = read_text(RUNNER_MAIN)

        self.assertIn('"[.exe] FILE.rdc [--outdir DIR] [--case NAME]"', runner)
        self.assertIn('argument == "--rdc"', runner)
        self.assertIn('argument == "--outdir" || argument == "--out-dir"', runner)
        self.assertIn('argument == "--case"', runner)
        self.assertIn('argument == "--width"', runner)
        self.assertIn('argument == "--height"', runner)
        self.assertIn('argument == "--trace-draw-actions"', runner)
        self.assertIn('"only one RDC input may be specified"', runner)
        self.assertIn(
            'Lower(PathToUtf8(options->rdc.extension())) != ".rdc"', runner
        )
        self.assertIn("Sha256File(options.rdc, &digest", runner)

        self.assertIn(
            "request.arguments = {PathToUtf8(options.rdc), PathToUtf8(png)}",
            runner,
        )
        self.assertIn(
            "player_request.arguments = {PathToUtf8(options.rdc)",
            runner,
        )
        self.assertIn('kResultSchema = "pvrgpu.backend-result.v1"', runner)
        self.assertIn('root / "backend-result.json"', runner)
        self.assertIn(
            "const std::filesystem::path artifact_root = options.outdir", runner
        )
        self.assertNotIn("legacy_png_outdir", runner)
        self.assertGreaterEqual(runner.count('options.outdir / "counter.txt"'), 1)
        self.assertGreaterEqual(runner.count('artifact_root / "counter.txt"'), 1)
        self.assertIn('options.outdir / "frame.png"', runner)
        self.assertIn('artifact_root / "frame.png"', runner)

    def test_llvmpipe_runner_preserves_mesa_counter_and_png_contract(self) -> None:
        runner = read_text(RUNNER_MAIN)
        patch = read_text(COUNTER_MESA_ZERO_OUTPUT_PATCH)

        for contract in (
            'environment["GALLIUM_DRIVER"] = "llvmpipe"',
            'environment["PVRGPU_RDC_SHA256"] = digest',
            'environment["LIBGL_DRIVERS_PATH"]',
            'environment["MESA_COUNTER_REPORT_PATH"] = PathToUtf8(report)',
            'environment["MESA_COUNTER_FRAME_SELECTION_MS"] = "replay"',
            'environment["RENDERDOC_MESA_EGL_PATH"]',
            'environment["RENDERDOC_MESA_GLES_PATH"]',
            'request.unset_environment = {"MESA_COUNTER_FRAME_TIME_MS"',
            'options.outdir / "Report.md"',
            'Lower(report_text).find("llvmpipe")',
            'ParseGoldenCounters(report_text',
            'FormatCounters(counters)',
        ):
            with self.subTest(contract=contract):
                self.assertIn(contract, runner)

        config = read_text(CONFIG_EXAMPLE)
        self.assertIn("PVRGPU_LLVMPIPE_MESA_PREFIX=", config)
        self.assertIn("PVRGPU_BUILD_DIR=", config)
        self.assertIn("PVRGPU_RDC_PVRGPU_RUNNER=", config)

        self.assertIn("draw_stats_clipper_primitives", patch)
        self.assertIn("prim_info->count == 0", patch)
        self.assertIn("!prim_info->primitive_lengths", patch)

    def test_pvrgpu_runner_forwards_driver_and_systemc_environment(self) -> None:
        runner = read_text(RUNNER_MAIN)

        for contract in (
            'environment["GALLIUM_DRIVER"] = "pvrgpu"',
            'environment["LIBGL_DRIVERS_PATH"] = PathToUtf8(driver_search)',
            'environment["MESA_GLES_VERSION_OVERRIDE"] = gles',
            'environment["PVRGPU_DRIVER_COMMAND_OUT"] = PathToUtf8(command)',
            'environment["PVRGPU_DRIVER_COUNTER_OUT"] = PathToUtf8(driver_counter)',
            'environment["PVRGPU_RDC_CASE_NAME"] = options.case_name',
            'environment["PVRGPU_RDC_OUTPUT_WIDTH"]',
            'environment["PVRGPU_RDC_OUTPUT_HEIGHT"]',
            'environment["PVRGPU_RDC_TRACE_DRAW_ACTIONS"]',
            'environment["PVRGPU_SYSTEMC_API_LIB"] = PathToUtf8(bridge)',
            'environment["PVRGPU_SYSTEMC_JSONL_OUT"] = PathToUtf8(model_stdout)',
            'environment["PVRGPU_SYSTEMC_STDERR_OUT"] = PathToUtf8(model_stderr)',
            'environment["PVRGPU_SYSTEMC_OUTDIR"] = PathToUtf8(model_png_dir)',
        ):
            with self.subTest(contract=contract):
                self.assertIn(contract, runner)

        self.assertIn('options.case_name.rfind("dEQP-GLES32.", 0)', runner)
        self.assertIn('options.case_name.rfind("dEQP-GLES31.", 0)', runner)
        self.assertIn('config.Get("PVRGPU_MESA_GLES_VERSION_OVERRIDE")', runner)
        self.assertIn('config.Path("PVRGPU_MESA_PVRGPU_PREFIX"', runner)
        self.assertIn('config.Path("PVRGPU_MODEL_STUB"', runner)
        self.assertIn('config.Path("PVRGPU_SYSTEMC_API_LIB")', runner)
        self.assertIn('config.Path("PVRGPU_SYSTEMC_BRIDGE")', runner)
        self.assertIn('artifact_root / "driver-command.txt"', runner)
        self.assertIn('artifact_root / "driver-counter.txt"', runner)
        self.assertIn('artifact_root / "model.stdout.jsonl"', runner)
        self.assertIn('artifact_root / "model.stderr.log"', runner)
        self.assertIn("pvrgpu.rdc-native-runner.v1", runner)
        self.assertIn("ValidatePvrgpuCompletion(bound_jsonl", runner)
        self.assertIn('type == "error"', runner)
        self.assertIn('"PvrGPU done message is missing pool_leaks"', runner)
        self.assertIn(
            '"PvrGPU model reported a non-zero MemoryPool leak count"', runner
        )

    def test_cmake_uses_compiler_appropriate_flags_and_exports_windows_bridge(
        self,
    ) -> None:
        cmake = read_text(CMAKE_FILE)

        self.assertIn("function(pvrgpu_target_warnings target)", cmake)
        self.assertIn("if(MSVC)", cmake)
        self.assertIn("target_compile_options(${target} PRIVATE /W4)", cmake)
        self.assertIn(
            "target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)",
            cmake,
        )
        self.assertIn("WINDOWS_EXPORT_ALL_SYMBOLS ON", cmake)

    def test_mesa_driver_seam_owns_the_previous_smoke_contracts(self) -> None:
        meson = read_text(DRIVER_ROOT / "meson.build")
        listed_sources = set(re.findall(r"'([^']+\.c)'", meson))
        actual_sources = {path.name for path in DRIVER_ROOT.glob("*.c")}
        self.assertEqual(actual_sources, listed_sources)
        self.assertIn("driver_pvrgpu = declare_dependency", meson)
        self.assertIn("-DGALLIUM_PVRGPU", meson)
        self.assertIn("idep_nir", meson)

        counter = read_text(DRIVER_ROOT / "pvrgpu_counter.c")
        counter_header = read_text(DRIVER_ROOT / "pvrgpu_counter.h")
        self.assertIn('getenv("PVRGPU_DRIVER_COUNTER_OUT")', counter)
        self.assertIn('"schema=%s producer=%s event=%s"', counter)
        self.assertIn('"pvrgpu.driver-counter.v1"', counter_header)

        event_contracts = {
            "pvrgpu_clear.c": (
                'pvrgpu_counter_eventf("clear_color"',
                'pvrgpu_counter_eventf("clear_depth_stencil"',
                "rgba=%u,%u,%u,%u",
            ),
            "pvrgpu_state.c": (
                'pvrgpu_counter_eventf("create_shader"',
                'pvrgpu_counter_eventf("set_vertex_buffers"',
                'pvrgpu_counter_eventf("create_blend_state"',
                'pvrgpu_counter_eventf("bind_blend_state"',
                'pvrgpu_counter_eventf("create_depth_stencil_alpha_state"',
                'pvrgpu_counter_eventf("bind_depth_stencil_alpha_state"',
                'pvrgpu_counter_eventf("create_rasterizer_state"',
                'pvrgpu_counter_eventf("bind_rasterizer_state"',
                'pvrgpu_counter_eventf("create_sampler_state"',
                'pvrgpu_counter_eventf("bind_sampler_states"',
                'pvrgpu_counter_eventf("create_sampler_view"',
                'pvrgpu_counter_eventf("set_sampler_views"',
                'pvrgpu_counter_eventf("set_constant_buffer"',
                '"has_words=%u',
            ),
            "pvrgpu_resource.c": (
                '"texture_map"',
                'pvrgpu_counter_eventf("texture_subdata"',
                'pvrgpu_counter_eventf("resource_copy_region"',
                'pvrgpu_counter_eventf("blit"',
            ),
            "pvrgpu_context.c": (
                'pvrgpu_counter_eventf("draw_triangles"',
                'pvrgpu_counter_eventf("draw_indexed_triangles"',
                'pvrgpu_counter_eventf("draw_textured_triangles"',
                'pvrgpu_counter_eventf("draw_uniform_triangles"',
                'pvrgpu_counter_eventf("set_framebuffer_state"',
                'pvrgpu_counter_eventf("flush"',
            ),
        }
        for filename, contracts in event_contracts.items():
            source = read_text(DRIVER_ROOT / filename)
            for contract in contracts:
                with self.subTest(source=filename, contract=contract):
                    self.assertIn(contract, source)

    def test_python_frontends_point_to_native_build_outputs(self) -> None:
        report_worker = read_text(PROJECT_ROOT / "tools" / "rdc_counter_report.py")
        deqp_worker = read_text(PROJECT_ROOT / "tools" / "deqp_capture_report.py")
        ui = read_text(PROJECT_ROOT / "tools" / "rdc_counter_ui.py")

        for worker in (report_worker, deqp_worker):
            self.assertIn('os.environ.get("PVRGPU_BUILD_DIR"', worker)
            self.assertIn('default_runner_path(work_root, "llvmpipe")', worker)
            self.assertIn('default_runner_path(work_root, "pvrgpu")', worker)

        self.assertIn(
            'DEFAULT_WORKER = PROJECT_ROOT / "tools" / "rdc_counter_report.py"',
            ui,
        )
        self.assertIn('os.environ.get("PVRGPU_BUILD_DIR"', ui)
        self.assertIn('f"llvmpipe{NATIVE_EXECUTABLE_SUFFIX}"', ui)
        self.assertIn('f"pvrgpu{NATIVE_EXECUTABLE_SUFFIX}"', ui)
        for source in (read_text(CMAKE_FILE), report_worker, deqp_worker, ui):
            self.assertNotIn("/bin/bash", source)


if __name__ == "__main__":
    unittest.main()
