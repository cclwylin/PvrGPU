from __future__ import annotations

import ast
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TextureMemoryPathConfigurationTests(unittest.TestCase):
    def test_control_ui_exposes_and_transports_independent_path(self) -> None:
        path = PROJECT_ROOT / "tools" / "pvrgpu_control.py"
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))

        self.assertIn('"Short · TPU → DRAM (default, fast)", "short"', source)
        self.assertIn('"Full cache · TPU → TCU → SLC → DRAM", "cached"', source)
        self.assertIn('self.settings.value("texture_memory_path", "short")', source)
        self.assertIn('"PVRGPU_TEXTURE_MEMORY_PATH", texture_memory_path', source)
        self.assertIn('"PVRGPU_MODEL_MEMORY_MODE",', source)
        self.assertIn('"bypass" if cache_bypass == "on" else "cache"', source)
        self.assertIn('environment.remove("PVRGPU_TEXTURE_MEMORY_PATH")', source)
        self.assertIn('"--texture-memory-path", texture_memory_path', source)
        self.assertIn("def _texture_memory_path_changed(self)", source)
        self.assertIn('self.cache_bypass_combo.findData("off")', source)

    def test_runner_wrapper_and_local_config_share_short_default(self) -> None:
        dynamic = (PROJECT_ROOT / "script" / "run_deqp_dynamic.sh").read_text(
            encoding="utf-8"
        )
        level = (PROJECT_ROOT / "script" / "run_deqp_level.sh").read_text(
            encoding="utf-8"
        )
        config = (PROJECT_ROOT / "config" / "local.env.example").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            'export PVRGPU_TEXTURE_MEMORY_PATH="${PVRGPU_TEXTURE_MEMORY_PATH:-short}"',
            dynamic,
        )
        self.assertIn("--texture-memory-path PATH", dynamic)
        self.assertIn(
            'PVRGPU_TEXTURE_MEMORY_PATH="${opt_texture_memory_path}"', dynamic
        )
        self.assertIn(
            '--texture-memory-path cached requires PVRGPU_MODEL_MEMORY_MODE=cache',
            dynamic,
        )
        self.assertIn(
            'echo "texture_memory_path=${PVRGPU_TEXTURE_MEMORY_PATH}"', dynamic
        )
        self.assertIn("PVRGPU_TEXTURE_MEMORY_PATH", level)
        self.assertIn("PVRGPU_TEXTURE_MEMORY_PATH=short", config)


if __name__ == "__main__":
    unittest.main()
