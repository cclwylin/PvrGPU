#!/usr/bin/env bash
set -euo pipefail

# Manhattan Frame 0 逐 DrawList 除錯啟動腳本
# 所有暫存檔案、輸出 PNG 與 Log 一律存放在 working root 外部。

WORK_DIR="${PVRGPU_SANDBOX_WORK_ROOT:-${HOME}/Downloads/_Codex/Working/SandBox}"
mkdir -p "${WORK_DIR}"

TARGET_DRAW_ID="${1:-0}"

echo "================================================================"
echo " PvrGPU GFXBench Manhattan Frame 0 DrawList Debugger"
echo " Target Prefix: DrawList[0 .. ${TARGET_DRAW_ID}]"
echo " Output Dir   : ${WORK_DIR}"
echo "================================================================"

# 輸出設定資訊到 Working 目錄
cat <<EOF > "${WORK_DIR}/debug_plan.json"
{
  "benchmark": "GFXBench Manhattan",
  "api": "OpenGL ES 3.0",
  "target_frame": 0,
  "target_draw_id": ${TARGET_DRAW_ID},
  "working_dir": "${WORK_DIR}"
}
EOF

echo "✓ 除錯環境與設定已就緒。暫存檔輸出位置：${WORK_DIR}"
