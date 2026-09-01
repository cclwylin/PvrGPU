from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from deqp_groups import GROUP_SPECS  # noqa: E402
from deqp_groups import exact_case_belongs_to_group  # noqa: E402
from deqp_groups import filter_exact_cases, get_group, is_exact_case  # noqa: E402


class DeqpGroupCatalogTests(unittest.TestCase):
    def test_catalog_has_exactly_24_stable_unique_groups(self) -> None:
        expected_ids = (
            "egl-create-context",
            "egl-image",
            "egl-robustness",
            "gles3-color-clear",
            "gles3-fbo",
            "gles3-fragment-ops",
            "gles3-instancing",
            "gles3-rasterization-primitives",
            "gles3-scissor",
            "gles3-shader-builtins",
            "gles3-texture-compressed",
            "gles3-texture-filtering",
            "gles3-transform-feedback",
            "gles3-ubo",
            "gles3-vertex-arrays",
            "gles3-stress-draw",
            "gles3-stress-memory",
            "gles3-stress-shaders",
            "gles31-compute-basic",
            "gles31-stress-draw-indirect",
            "gles31-ssbo",
            "gles31-texture-multisample",
            "gles32-geometry-shading",
            "gles32-tessellation",
        )

        self.assertEqual(len(GROUP_SPECS), 24)
        self.assertEqual(len({group.id for group in GROUP_SPECS}), 24)
        self.assertEqual(tuple(group.id for group in GROUP_SPECS), expected_ids)
        self.assertTrue(all(group.label for group in GROUP_SPECS))
        self.assertTrue(all(group.selectors for group in GROUP_SPECS))
        self.assertEqual(sum(group.locked_case_count for group in GROUP_SPECS), 16_608)

    def test_locked_cts_renames_are_canonical(self) -> None:
        self.assertEqual(
            get_group("gles3-instancing").selectors,
            ("dEQP-GLES3.functional.instanced.*",),
        )
        self.assertEqual(
            get_group("gles3-scissor").selectors,
            ("dEQP-GLES3.functional.fragment_ops.scissor.*",),
        )
        self.assertEqual(
            get_group("gles3-stress-shaders").selectors,
            (
                "dEQP-GLES3.stress.long_shaders.*",
                "dEQP-GLES3.stress.long_running_shaders.*",
            ),
        )
        self.assertEqual(
            get_group("gles31-stress-draw-indirect").selectors,
            ("dEQP-GLES31.stress.draw_indirect.*",),
        )
        self.assertEqual(get_group("gles31-stress-draw-indirect").locked_case_count, 23)

    def test_gles32_display_aliases_use_the_gles31_runner_root(self) -> None:
        geometry = get_group("gles32-geometry-shading")
        tessellation = get_group("gles32-tessellation")

        self.assertEqual(geometry.suite, "dEQP-GLES31")
        self.assertEqual(
            geometry.selectors,
            ("dEQP-GLES31.functional.geometry_shading.*",),
        )
        self.assertEqual(tessellation.suite, "dEQP-GLES31")
        self.assertEqual(
            tessellation.selectors,
            ("dEQP-GLES31.functional.tessellation.*",),
        )
        self.assertTrue(
            exact_case_belongs_to_group(
                geometry,
                "dEQP-GLES31.functional.geometry_shading.basic.points",
            )
        )
        self.assertFalse(
            exact_case_belongs_to_group(
                geometry,
                "dEQP-GLES32.functional.geometry_shading.basic.points",
            )
        )

    def test_mislabeled_egl_robustness_group_uses_egl_cases(self) -> None:
        robustness = get_group("egl-robustness")

        self.assertEqual(robustness.suite, "dEQP-EGL")
        self.assertTrue(
            exact_case_belongs_to_group(
                robustness,
                "dEQP-EGL.functional.robustness.create_context.lose_context_on_reset",
            )
        )
        self.assertTrue(
            exact_case_belongs_to_group(
                robustness,
                "dEQP-EGL.functional.get_proc_address.extension.gl_khr_robustness",
            )
        )
        self.assertFalse(
            exact_case_belongs_to_group(
                robustness,
                "dEQP-GLES31.functional.robustness.buffer_access",
            )
        )

    def test_filter_returns_only_unique_exact_cases_for_the_group(self) -> None:
        first = "dEQP-GLES3.functional.color_clear.single_rgb"
        second = "dEQP-GLES3.functional.color_clear.single_rgba"
        cases = filter_exact_cases(
            "gles3-color-clear",
            (
                first,
                "dEQP-GLES3.functional.color_clear.*",
                first,
                second,
                "dEQP-GLES3.functional.fbo.complete",
                "dEQP-GLES3.functional.color_clear.single_rgb,other",
                "TEST: dEQP-GLES3.functional.color_clear.masked_rgb",
            ),
        )

        self.assertEqual(cases, (first, second))
        self.assertTrue(all(is_exact_case(case_name) for case_name in cases))

    def test_availability_reason_tracks_the_current_runtime_limit(self) -> None:
        self.assertTrue(get_group("egl-image").available)
        self.assertIsNone(get_group("egl-image").availability_reason)
        self.assertFalse(get_group("gles3-fbo").available)
        self.assertIn("EGL_OPENGL_ES3_BIT_KHR", get_group("gles3-fbo").availability_reason or "")
        self.assertFalse(get_group("gles32-tessellation").available)


if __name__ == "__main__":
    unittest.main()
