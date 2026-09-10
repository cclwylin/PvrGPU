import unittest
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from script.deqp_4level_catalog import (
    DEFAULT_GL_CONFIG,
    GROUPS_BY_NUMBER,
    TIERS_BY_ID,
    build_plan,
    is_long_only_case,
    plan_from_cases,
)


class DeqpFourLevelLongPolicyTest(unittest.TestCase):
    def setUp(self):
        self.ordinary = "dEQP-GLES3.functional.multisample.fbo_4_samples.depth"
        self.max_constancy = (
            "dEQP-GLES3.functional.multisample.fbo_max_samples."
            "constancy_alpha_to_coverage"
        )
        self.stress = "dEQP-GLES3.stress.long_running_shaders.infinite_for_fragment"
        self.slow_texture = (
            "dEQP-GLES31.functional.texture.multisample.samples_8."
            "sample_mask_and_alpha_to_coverage"
        )
        self.slow_image = (
            "dEQP-GLES31.functional.image_load_store.3d.atomic."
            "comp_swap_r32ui_return_value"
        )
        self.slow_draw_arrays = (
            "dEQP-GLES31.stress.draw_indirect.drawarrays."
            "data_over_bounds_with_primcount"
        )
        self.slow_draw_elements = (
            "dEQP-GLES31.stress.draw_indirect.drawelements."
            "data_over_bounds_with_primcount"
        )
        self.discovery = {"gles3": [self.ordinary, self.max_constancy, self.stress]}

    def test_catalog_totals(self):
        self.assertEqual(
            {tier: TIERS_BY_ID[tier].cases for tier in ("L1", "L2", "L3", "L4")},
            {"L1": 2474, "L2": 7750, "L3": 30224, "L4": 90360},
        )

    def test_l1_l3_filter_before_sampling(self):
        for tier_id in ("L1", "L2", "L3"):
            tier = TIERS_BY_ID[tier_id]
            plan = build_plan(
                name=tier_id,
                tier=tier,
                groups=[GROUPS_BY_NUMBER[18], GROUPS_BY_NUMBER[30]],
                discovery=self.discovery,
                shards=1,
                log_images="disable",
                default_config=DEFAULT_GL_CONFIG,
            )
            cases = [case for bucket in plan.buckets for case in bucket.cases]
            self.assertEqual(cases, [self.ordinary])
            self.assertEqual(plan.long_excluded_count, 2)
            self.assertFalse(any(is_long_only_case(case) for case in cases))

    def test_l4_and_explicit_debug_keep_long_cases(self):
        tier = TIERS_BY_ID["L4"]
        l4 = build_plan(
            name="L4",
            tier=tier,
            groups=[],
            discovery=self.discovery,
            shards=1,
            log_images="enable",
            default_config=DEFAULT_GL_CONFIG,
            whole_modules=True,
        )
        explicit = plan_from_cases(
            name="debug",
            tier=TIERS_BY_ID["L1"],
            cases=[self.max_constancy, self.stress],
            shards=1,
            log_images="disable",
            default_config=DEFAULT_GL_CONFIG,
        )
        self.assertEqual(l4.total_cases, 3)
        self.assertEqual(explicit.total_cases, 2)

    def test_measured_multi_minute_cases_are_l4_only(self):
        self.assertTrue(is_long_only_case(self.slow_texture))
        self.assertTrue(is_long_only_case(self.slow_image))
        self.assertTrue(is_long_only_case(self.slow_draw_arrays))
        self.assertTrue(is_long_only_case(self.slow_draw_elements))


if __name__ == "__main__":
    unittest.main()
