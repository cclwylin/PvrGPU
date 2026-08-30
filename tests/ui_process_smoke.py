"""Offscreen QProcess integration test for the PvrGPU Qt controller."""

from __future__ import annotations

import argparse

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from pvrgpu_control import GLBENCH_CASES, SYSTEMC_FUNCTIONAL_CASES, MainWindow


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("llvmpipe", "pvrgpu"), required=True)
    parser.add_argument("--case", default="fill_solid")
    parser.add_argument("--cache-bypass", choices=("off", "on"), default="off")
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--timeout-ms", type=int, default=15000)
    options = parser.parse_args()

    advertised_cases = tuple(case_name for _, _, case_name in GLBENCH_CASES)
    if len(advertised_cases) != 20 or len(set(advertised_cases)) != 20:
        raise RuntimeError("GLBench selector must advertise exactly 20 unique cases")
    if SYSTEMC_FUNCTIONAL_CASES != frozenset(advertised_cases):
        raise RuntimeError(
            "SystemC functional support must cover all advertised GLBench cases"
        )

    if options.width <= 0 or options.height <= 0:
        parser.error("--width and --height must be positive")
    if options.timeout_ms <= 0:
        parser.error("--timeout-ms must be positive")

    app = QApplication([])
    window = MainWindow()
    window.backend_combo.setCurrentIndex(window.backend_combo.findData(options.backend))
    cache_index = window.cache_bypass_combo.findData(options.cache_bypass)
    if cache_index < 0:
        raise RuntimeError(f"{options.cache_bypass} is missing from cache bypass selector")
    window.cache_bypass_combo.setCurrentIndex(cache_index)
    for case_index in range(window.case_combo.count()):
        if window.case_combo.itemData(case_index)[1] == options.case:
            window.case_combo.setCurrentIndex(case_index)
            break
    else:
        raise RuntimeError(f"{options.case} is missing from the case selector")
    window.samples_spin.setValue(2 if options.backend == "pvrgpu" else 1)
    window.width_spin.setValue(options.width)
    window.height_spin.setValue(options.height)
    result = {"code": 5}

    def finished(*_args: object) -> None:
        window._poll_artifacts()
        command_log = window.log.toPlainText()
        cache_control_ok = window.cache_bypass_combo.isEnabled() == (
            options.backend == "pvrgpu"
        )
        if options.backend == "llvmpipe":
            ok = (
                bool(window.records)
                and window.renderer_verified
                and window.latest_image is not None
                and cache_control_ok
                and "--cache-bypass" not in command_log
            )
        else:
            pngs = sorted((window.run_dir / "png").glob("*.png")) if window.run_dir else []
            texture_path_ok = True
            texture_taps = {
                "fill_tex_nearest": 1,
                "fill_tex_bilinear": 4,
                "fill_tex_trilinear_linear_01": 8,
                "fill_tex_trilinear_linear_04": 8,
                "fill_tex_trilinear_linear_05": 8,
            }
            if options.case in texture_taps:
                taps_per_request = texture_taps[options.case]
                texture_path_ok = all(
                    record.values.get("texture_requests", 0) > 0
                    and record.values.get("texel_fetches", 0)
                    == record.values.get("texture_requests", 0)
                    * taps_per_request
                    and record.values.get("tcu_line_accesses", 0)
                    == record.values.get("texel_fetches", 0)
                    and record.values.get("tcu_read_accesses", 0)
                    == record.values.get("texel_fetches", 0)
                    and record.values.get("tcu_cycles", 0)
                    == record.values.get("texel_fetches", 0)
                    for record in window.records
                )
            if options.cache_bypass == "on":
                memory_path_ok = all(
                    record.values.get("slc_line_accesses") == 0
                    and record.values.get("slc_bypassed") == 1
                    and record.values.get("dram_write_transactions", 0) > 0
                    and record.values.get("dram_read_transactions", 0) > 0
                    for record in window.records
                )
            else:
                memory_path_ok = all(
                    record.values.get("slc_line_accesses", 0) > 0
                    and record.values.get("slc_bypassed") == 0
                    and record.values.get("dram_write_transactions", 0) > 0
                    and record.values.get("dram_read_transactions", 0) > 0
                    for record in window.records
                )
            ok = (
                len(window.records) == 2
                and all(record.provenance == "modeled" for record in window.records)
                and all(record.values.get("virtual_gpu_cycles", 0) > 0 for record in window.records)
                and window.latest_image is not None
                and window.latest_image.is_file()
                and "functional" in window.renderer_value.text().lower()
                and len(pngs) == 2
                and all(len(record.drawlist_stats) == 1 for record in window.records)
                and window.drawlist_table.rowCount() == 2
                and cache_control_ok
                and f"--cache-bypass {options.cache_bypass}" in command_log
                and memory_path_ok
                and texture_path_ok
            )
        print(
            "UI_PROCESS_SMOKE",
            options.backend,
            options.case,
            f"cache_bypass={options.cache_bypass}",
            "PASS" if ok else "FAIL",
            f"records={len(window.records)}",
            f"renderer={window.renderer_value.text()}",
            f"latest_image={window.latest_image}",
        )
        result["code"] = 0 if ok else 3
        app.quit()

    window.process.finished.connect(finished)
    QTimer.singleShot(0, window._start)
    QTimer.singleShot(options.timeout_ms, app.quit)
    app.exec()
    return result["code"]


if __name__ == "__main__":
    raise SystemExit(main())
