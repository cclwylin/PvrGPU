#!/usr/bin/env python3
"""Compare two RGBA8 PNGs after decoding, independent of PNG compression."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from check_model_pipeline import decode_rgba8_png  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} GOLDEN.png MODEL.png", file=sys.stderr)
        return 2

    golden_path = Path(sys.argv[1]).resolve()
    model_path = Path(sys.argv[2]).resolve()
    golden_width, golden_height, golden = decode_rgba8_png(golden_path)
    model_width, model_height, model = decode_rgba8_png(model_path)
    if (golden_width, golden_height) != (model_width, model_height):
        print(
            "PNG dimension mismatch: "
            f"golden={golden_width}x{golden_height}, "
            f"model={model_width}x{model_height}",
            file=sys.stderr,
        )
        return 1

    differing_pixels = 0
    max_channel_delta = 0
    first_difference: tuple[int, int, bytes, bytes] | None = None
    for pixel in range(golden_width * golden_height):
        offset = pixel * 4
        golden_pixel = golden[offset : offset + 4]
        model_pixel = model[offset : offset + 4]
        if golden_pixel == model_pixel:
            continue
        differing_pixels += 1
        max_channel_delta = max(
            max_channel_delta,
            *(abs(first - second) for first, second in zip(golden_pixel, model_pixel)),
        )
        if first_difference is None:
            first_difference = (
                pixel % golden_width,
                pixel // golden_width,
                golden_pixel,
                model_pixel,
            )

    if first_difference is not None:
        x, y, golden_pixel, model_pixel = first_difference
        print(
            f"RGBA_MISMATCH pixels={differing_pixels} "
            f"max_channel_delta={max_channel_delta} first=({x},{y}) "
            f"golden={tuple(golden_pixel)} model={tuple(model_pixel)}",
            file=sys.stderr,
        )
        return 1

    print(
        f"RGBA_MATCH size={golden_width}x{golden_height} "
        "differing_pixels=0 max_channel_delta=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
