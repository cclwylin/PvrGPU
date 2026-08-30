from __future__ import annotations

from pathlib import Path
import os
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "install-pvrgpu-mesa-driver.sh"
SMOKE_SCRIPT = PROJECT_ROOT / "scripts" / "run-pvrgpu-mesa-driver-smoke.sh"
TRIANGLE_SMOKE_SCRIPT = (
    PROJECT_ROOT / "scripts" / "run-pvrgpu-mesa-driver-triangle-smoke.sh"
)
PHASE3_SMOKE_SCRIPT = (
    PROJECT_ROOT / "scripts" / "run-pvrgpu-mesa-driver-phase3-state-smoke.sh"
)
PHASE4_SMOKE_SCRIPT = (
    PROJECT_ROOT / "scripts" / "run-pvrgpu-mesa-driver-phase4-texture-smoke.sh"
)
PHASE5_SMOKE_SCRIPT = (
    PROJECT_ROOT / "scripts" / "run-pvrgpu-mesa-driver-phase5-fbo-smoke.sh"
)
PHASE6_SMOKE_SCRIPT = (
    PROJECT_ROOT / "scripts" / "run-pvrgpu-mesa-driver-phase6-uniform-smoke.sh"
)
RDC_SYSTEMC_RUNNER = (
    PROJECT_ROOT / "scripts" / "run-rdc-pvrgpu-driver-systemc.sh"
)


class MesaIntegrationScriptTests(unittest.TestCase):
    def test_installer_patches_fake_mesa_tree_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mesa = Path(tmp) / "mesa"
            self._write_fake_mesa_tree(mesa)

            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            for _ in range(2):
                subprocess.run(
                    [str(SCRIPT), "--mesa-src", str(mesa)],
                    cwd=PROJECT_ROOT,
                    env=env,
                    check=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            subprocess.run(
                [str(SCRIPT), "--mesa-src", str(mesa), "--check"],
                cwd=PROJECT_ROOT,
                env=env,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertTrue(
                (mesa / "src" / "gallium" / "drivers" / "pvrgpu" / "pvrgpu_cmd.h").is_file()
            )
            self.assertIn("'pvrgpu'", (mesa / "meson.options").read_text())
            self.assertIn(
                "with_gallium_pvrgpu = gallium_drivers.contains('pvrgpu')",
                (mesa / "meson.build").read_text(),
            )
            self.assertIn(
                "subdir('drivers/pvrgpu')",
                (mesa / "src" / "gallium" / "meson.build").read_text(),
            )
            self.assertIn(
                "driver_swrast, driver_pvrgpu, driver_r300",
                (mesa / "src" / "gallium" / "targets" / "dri" / "meson.build").read_text(),
            )
            sw_helper = (
                mesa
                / "src"
                / "gallium"
                / "auxiliary"
                / "target-helpers"
                / "sw_helper.h"
            ).read_text()
            self.assertIn('#include "pvrgpu/pvrgpu_public.h"', sw_helper)
            self.assertIn('strcmp(driver, "pvrgpu") == 0', sw_helper)
            self.assertNotIn('"pvrgpu",\n#endif\n#if defined(GALLIUM_LLVMPIPE)', sw_helper)

    def test_smoke_runner_checks_counter_channel(self) -> None:
        smoke = SMOKE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("PVRGPU_DRIVER_COUNTER_OUT", smoke)
        self.assertIn("driver-counter.txt", smoke)
        self.assertIn("CLEAR_PIXEL=32,64,128,255", smoke)
        self.assertIn("event=clear_color", smoke)
        self.assertIn("event=texture_map", smoke)

    def test_triangle_smoke_runner_checks_draw_counter_channel(self) -> None:
        smoke = TRIANGLE_SMOKE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("PVRGPU_DRIVER_COUNTER_OUT", smoke)
        self.assertIn("driver-counter.txt", smoke)
        self.assertIn("glDrawArrays(GL_TRIANGLES, 0, 3)", smoke)
        self.assertIn("glVertexAttribPointer", smoke)
        self.assertIn("TRIANGLE_DRAW=done", smoke)
        self.assertIn("event=create_shader stage=vertex", smoke)
        self.assertIn("event=create_shader stage=fragment", smoke)
        self.assertIn("event=set_vertex_buffers", smoke)
        self.assertIn("event=draw_triangles", smoke)

    def test_phase3_state_smoke_runner_checks_fixed_function_counters(self) -> None:
        smoke = PHASE3_SMOKE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("PVRGPU_DRIVER_COUNTER_OUT", smoke)
        self.assertIn("driver-counter.txt", smoke)
        self.assertIn("glEnable(GL_SCISSOR_TEST)", smoke)
        self.assertIn("glEnable(GL_CULL_FACE)", smoke)
        self.assertIn("glEnable(GL_DEPTH_TEST)", smoke)
        self.assertIn("glEnable(GL_BLEND)", smoke)
        self.assertIn("glColorMask", smoke)
        self.assertIn("glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_SHORT, indices)", smoke)
        self.assertIn("PHASE3_STATE_DRAW=done", smoke)
        self.assertIn("event=create_blend_state", smoke)
        self.assertIn("event=bind_blend_state", smoke)
        self.assertIn("event=create_depth_stencil_alpha_state", smoke)
        self.assertIn("event=bind_depth_stencil_alpha_state", smoke)
        self.assertIn("event=create_rasterizer_state", smoke)
        self.assertIn("event=bind_rasterizer_state", smoke)
        self.assertIn("event=set_blend_color", smoke)
        self.assertIn("event=set_stencil_ref", smoke)
        self.assertIn("event=draw_indexed_triangles", smoke)

    def test_phase4_texture_smoke_runner_checks_texture_counters(self) -> None:
        smoke = PHASE4_SMOKE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("PVRGPU_DRIVER_COUNTER_OUT", smoke)
        self.assertIn("driver-counter.txt", smoke)
        self.assertIn("glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 2, 2, 0", smoke)
        self.assertIn("GL_TEXTURE_MIN_FILTER, GL_NEAREST", smoke)
        self.assertIn("GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE", smoke)
        self.assertIn("uniform sampler2D u_texture", smoke)
        self.assertIn("texture2D(u_texture, v_texcoord)", smoke)
        self.assertIn("glDrawArrays(GL_TRIANGLES, 0, 3)", smoke)
        self.assertIn("PHASE4_TEXTURE_DRAW=done", smoke)
        self.assertIn("event=texture_subdata", smoke)
        self.assertIn("event=create_sampler_state", smoke)
        self.assertIn("event=bind_sampler_states", smoke)
        self.assertIn("event=create_sampler_view", smoke)
        self.assertIn("event=set_sampler_views", smoke)
        self.assertIn("event=draw_textured_triangles", smoke)

    def test_phase5_fbo_smoke_runner_checks_framebuffer_and_flush_counters(self) -> None:
        smoke = PHASE5_SMOKE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("PVRGPU_DRIVER_COUNTER_OUT", smoke)
        self.assertIn("driver-counter.txt", smoke)
        self.assertIn("glGenFramebuffers", smoke)
        self.assertIn("glFramebufferTexture2D", smoke)
        self.assertIn("glCheckFramebufferStatus(GL_FRAMEBUFFER)", smoke)
        self.assertIn("FBO_STATUS=complete", smoke)
        self.assertIn("FBO_CLEAR_PIXEL=64,128,191,255", smoke)
        self.assertIn("FBO_COPY_PIXEL=64,128,191,255", smoke)
        self.assertIn("glReadPixels", smoke)
        self.assertIn("glCopyTexSubImage2D", smoke)
        self.assertIn("glDrawArrays(GL_TRIANGLES, 0, 3)", smoke)
        self.assertIn("glFlush()", smoke)
        self.assertIn("glFinish()", smoke)
        self.assertIn("PHASE5_FBO_DRAW=done", smoke)
        self.assertIn("event=set_framebuffer_state", smoke)
        self.assertIn(
            "event=clear_color width=${fbo_width} height=${fbo_height}",
            smoke,
        )
        self.assertIn("event=resource_copy_region", smoke)
        self.assertIn("event=blit", smoke)
        self.assertIn("event=draw_triangles", smoke)
        self.assertIn("event=flush", smoke)

    def test_phase6_uniform_smoke_runner_checks_constant_buffer_counters(self) -> None:
        smoke = PHASE6_SMOKE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("PVRGPU_DRIVER_COUNTER_OUT", smoke)
        self.assertIn("driver-counter.txt", smoke)
        self.assertIn("uniform vec4 u_color[8]", smoke)
        self.assertIn("glUniform4fv", smoke)
        self.assertIn("glDrawArrays(GL_TRIANGLES, 0, 3)", smoke)
        self.assertIn("PHASE6_UNIFORM_DRAW=done", smoke)
        self.assertIn("event=set_constant_buffer stage=fragment", smoke)
        self.assertIn("has_buffer=1", smoke)
        self.assertIn("has_words=1", smoke)
        self.assertIn("event=draw_uniform_triangles", smoke)

    def test_rdc_runner_passes_case_name_to_driver(self) -> None:
        runner = RDC_SYSTEMC_RUNNER.read_text(encoding="utf-8")
        self.assertIn("PVRGPU_RDC_CASE_NAME", runner)
        self.assertIn("PVRGPU_RDC_OUTPUT_WIDTH", runner)
        self.assertIn("PVRGPU_RDC_OUTPUT_HEIGHT", runner)
        self.assertIn("dEQP-GLES31.*", runner)
        self.assertIn('default_gles_override="3.1"', runner)

    def _write_fake_mesa_tree(self, mesa: Path) -> None:
        (mesa / "src" / "gallium" / "targets" / "dri").mkdir(parents=True)
        (mesa / "src" / "gallium" / "auxiliary" / "target-helpers").mkdir(parents=True)

        (mesa / "meson.options").write_text(
            """
option(
  'gallium-drivers',
  type : 'array',
  value : ['auto'],
  choices : [
    'all', 'auto',
    'asahi', 'crocus', 'd3d12', 'ethosu', 'etnaviv', 'freedreno', 'i915', 'iris',
    'lima', 'llvmpipe', 'nouveau', 'panfrost', 'r300', 'r600', 'radeonsi',
    'rocket', 'softpipe', 'svga', 'tegra', 'v3d', 'vc4', 'virgl', 'zink',
  ],
)
""".strip()
            + "\n",
            encoding="utf-8",
        )
        (mesa / "meson.build").write_text(
            """
elif gallium_drivers.contains('all')
   gallium_drivers = [
     'r300', 'r600', 'radeonsi', 'crocus', 'v3d', 'vc4', 'freedreno', 'etnaviv', 'i915',
     'nouveau', 'svga', 'tegra', 'virgl', 'lima', 'panfrost', 'llvmpipe', 'softpipe', 'iris',
     'zink', 'd3d12', 'asahi', 'rocket', 'ethosu'
   ]
endif

with_gallium_panfrost = gallium_drivers.contains('panfrost')
with_gallium_etnaviv = gallium_drivers.contains('etnaviv')
""".strip()
            + "\n",
            encoding="utf-8",
        )
        (mesa / "src" / "gallium" / "meson.build").write_text(
            """
if with_any_llvmpipe and with_gallium_softpipe
  driver_swrast = declare_dependency(
    dependencies : [ driver_softpipe, driver_llvmpipe ],
  )
elif with_any_llvmpipe
  driver_swrast = driver_llvmpipe
elif with_gallium_softpipe
  driver_swrast = driver_softpipe
else
  driver_swrast = declare_dependency()
endif
""".strip()
            + "\n",
            encoding="utf-8",
        )
        (mesa / "src" / "gallium" / "targets" / "dri" / "meson.build").write_text(
            """
dependencies : [
  driver_swrast, driver_r300, driver_r600, driver_radeonsi, driver_nouveau,
]
""".strip()
            + "\n",
            encoding="utf-8",
        )
        (
            mesa
            / "src"
            / "gallium"
            / "auxiliary"
            / "target-helpers"
            / "sw_helper.h"
        ).write_text(
            """
#ifdef GALLIUM_D3D12
#include "d3d12/d3d12_public.h"
#endif

static inline struct pipe_screen *
sw_screen_create_named(struct sw_winsys *winsys, const struct pipe_screen_config *config, const char *driver)
{
   struct pipe_screen *screen = NULL;

#if defined(GALLIUM_LLVMPIPE)
   if (screen == NULL && (strcmp(driver, "llvmpipe") == 0 || !driver[0]))
      screen = llvmpipe_create_screen(winsys);
#endif

   return screen;
}
""".strip()
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
