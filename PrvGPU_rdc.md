# PrvGPU RDC Counter 測試

> 日期：2026-08-30  
> DUT：PvrGPU（SystemC）  
> 輸入：`$HOME/Downloads/_Codex/Working/drc_patterns/1.GLBench`  
> 原則：一個 `.rdc` 是一個 frame；只比較 counter，不比較 PNG。

## 測試流程

每個 `.rdc` 固定執行：

1. `RenderDoc + Mesa + llvmpipe`，產生 `counter_golden.txt`。
2. `RenderDoc + Mesa/POC + PvrGPU(SystemC)`，產生 `counter_pvrgpu.txt`。
3. 兩個文字檔做 byte-for-byte exact compare。
4. 完全相同：此 frame PASS，繼續下一個 `.rdc`。
5. 有差異：立即停止，只 debug PvrGPU；修正並讓此 frame PASS 後才繼續。

```text
.rdc
 ├─ Golden: RenderDoc + Mesa + llvmpipe ──> counter_golden.txt
 └─ DUT:    RenderDoc + Mesa/POC + PvrGPU ─> counter_pvrgpu.txt
                                                │
                                      exact text compare
                                      ├─ same: PASS → next
                                      └─ diff: STOP → debug PvrGPU
```

PNG 可以被底層程式順帶產生，但不參與 PASS/FAIL。

## Counter 格式

兩個檔案都固定為相同順序的 17 行 `name=value`：

```text
ia_vertices
ia_primitives
vs_invocations
gs_invocations
gs_primitives
c_invocations
c_primitives
ps_invocations
hs_invocations
ds_invocations
cs_invocations
ts_invocations
ms_invocations
ms_primitives
drawlists
setup_triangles
texel_fetches
```

正規化工具：`tools/rdc/write_counter_txt.py`

- Golden 輸入：RenderDoc/Mesa replay 的 `Report.md`
- PvrGPU 輸入：`pvrgpu.counter.v1` JSONL stdout
- 缺 frame、重複 counter、protocol error、非 PvrGPU backend、未完成或 pool leak 都直接失敗
- 正式 Mesa/POC runner 還會自動啟用 `--require-mesa-ingest`，缺少或偽造 command-ingest provenance 時，在 counter compare 前直接失敗

## 建置 Mesa/POC

Mesa 固定為 26.2.1，套用 `patches/mesa-26.2.1-pvrgpu-rdc-poc.patch`：

```bash
./scripts/build-rdc-poc-mesa.sh
```

預設安裝位置：

```text
$PVRGPU_WORK_ROOT/mesa-poc/install
```

這支 wrapper 可重複執行；已套用 patch 時不會重複套用。

## 正式 Batch

正式測試必須指定已實作的 `RenderDoc + Mesa/POC → PvrGPU` runner。執行單一 case：

```bash
PVRGPU_RDC_DUT_RUNNER="${PWD}/scripts/run-rdc-pvrgpu-poc.sh" \
  ./scripts/run-rdc-counter-batch.sh --case fill_solid
```

依 manifest 順序執行全部 frame：

```bash
PVRGPU_RDC_DUT_RUNNER="${PWD}/scripts/run-rdc-pvrgpu-poc.sh" \
  ./scripts/run-rdc-counter-batch.sh --all
```

從指定 frame 繼續（以下以 Frame 9 為例）：

```bash
PVRGPU_RDC_DUT_RUNNER="${PWD}/scripts/run-rdc-pvrgpu-poc.sh" \
  ./scripts/run-rdc-counter-batch.sh --start 9
```

Batch 會依 frozen manifest 順序逐一驗證 `.rdc` SHA-256、生成兩個 counter txt，並做 byte-for-byte compare。第一個 replay、ingest、protocol 或 counter 錯誤都會立即停止，不會跳到下一個 `.rdc`。

若不設定 `PVRGPU_RDC_DUT_RUNNER`，Batch 仍可使用舊的 `pvrgpu-model-stub --case` adapter 做內建 fixture 回歸；它不是正式 `.rdc → Mesa/POC → PvrGPU` 測試。

## 單一 RDC runner

`scripts/run-rdc-pvrgpu-poc.sh` 的固定介面為：

```bash
./scripts/run-rdc-pvrgpu-poc.sh \
  --rdc FILE.rdc \
  --case CASE \
  --width WIDTH \
  --height HEIGHT \
  --outdir DIRECTORY
```

以 Frame 1 為例：

```bash
manual_root="${PVRGPU_WORK_ROOT:-$HOME/Downloads/_Codex/Working/PvrGPU}/out/manual-fill-solid"
mkdir -p "${manual_root}"

./scripts/run-rdc-pvrgpu-poc.sh \
  --rdc "$HOME/Downloads/_Codex/Working/drc_patterns/1.GLBench/fill_solid/recorder/trace/fill_solid_capture_1.rdc" \
  --case fill_solid \
  --width 512 \
  --height 512 \
  --outdir "${manual_root}/png" \
  >"${manual_root}/stdout.jsonl" \
  2>"${manual_root}/stderr.log"
```

Runner 對一個 `.rdc` 做三件事：

1. 以 RenderDoc player 和 Mesa/POC llvmpipe replay，生成 RenderDoc API trace 與 Gallium trace。
2. `tools/rdc/build_mesa_poc_command.py` 將兩份 trace 轉成嚴格的 `pvrgpu.mesa-poc-command.v1` capsule。
3. 以 `pvrgpu-model-stub --mesa-command command.txt` 執行 SystemC，stdout 只交付 PvrGPU JSONL。

中間證據位於 `outdir` 同層的 `mesa-poc/`：

```text
pvrgpu/
├── stdout.jsonl
├── stderr.log
├── png/                    # 可生成，但不比較
└── mesa-poc/
    ├── renderdoc-api.md
    ├── gallium.xml
    ├── command.txt
    ├── replay.png          # 不參與 PASS/FAIL
    ├── player.stdout.log
    ├── player.stderr.log
    └── translator.stderr.log
```

## 嚴格 Mesa ingest provenance

正式 external runner 模式下，Batch 呼叫 counter 正規化工具時會自動加入 `--require-mesa-ingest`。PvrGPU 的唯一一筆 `hello` 必須同時包含：

```text
backend=pvrgpu
mesa_command_ingest=true
command_source=renderdoc-mesa-gallium-trace-poc
mesa_command_schema=pvrgpu.mesa-poc-command.v1
rdc_sha256=<64 位 lowercase hex>
api_trace_sha256=<64 位 lowercase hex>
gallium_trace_sha256=<64 位 lowercase hex>
```

因此內建 fixture adapter 的 `mesa_command_ingest=false` 不能混入正式結果。Capsule 另會驗證 manifest case、尺寸、RDC hash、API draw/shader 及 Gallium command/state/buffer 證據；驗證失敗時 SystemC 不會啟動。

## Trace ingest 範圍與界線

正式 20-frame run 不只檢查 trace 的存在；capsule 會把每個 case 實際 replay 的命令、buffer 與 state 納入 fail-closed 驗證：

- Multi-attribute（Frame 8～11）：保留 Gallium vertex-element 數量、`vertex_buffer_map` 與實際 VBO/index bytes；多個 attribute 指向同一 resource 時，不複製成虛構的獨立資料。
- Varying（Frame 12～15）：保留實際 VBO/index bytes、API shader hash 與 draw state，再映射到已固定驗證的 PCO shader及 VS→FS varying linkage。
- Texture（Frame 16～20）：ingest 兩個實際 VBO（position、texcoord）、VS constant、RGBA8 512×512 至 1×1 的 10 個 mip actual upload bytes，以及實際 sampler filter state。Texture payload 來自 Gallium trace，不以 builtin generator 代替。
- 所有 DSA、blend、rasterizer、vertex-elements、framebuffer、constant、VBO 與 sampler/view state，都以 selected draw 的 `pipe` 為界，依 draw 前的 create→bind→delete/set 時序解析；不會拿其它 context 或最後建立但未綁定的 state 代替。
- PvrGPU 內部 GPU address、Rogue texture/sampler descriptors、PCO binary 與 varying register linkage，則在 trace contract 完全符合後做固定且可驗證的 exact semantic mapping；不會把 llvmpipe 的 process-local pointer 當成 PvrGPU address。

這一層是只接受 frozen manifest 20 個 workload contract 的 **trace capsule adapter**，不是 production Gallium driver。它不支援任意 Gallium shader、resource、draw 或 state，也不代表 Mesa/RenderDoc 已能把 PvrGPU 當成一般驅動直接使用；不在已驗證 contract 內的輸入一律拒絕。

## Batch 產物

每個 case 的主要輸出是：

```text
NNN-case/
├── counter_golden.txt
├── counter_pvrgpu.txt
├── counter_diff.txt       # 只有 mismatch 才產生
├── result.txt
├── golden/
└── pvrgpu/
```

## 目前結果

正式 `renderdoc-mesa-poc-runner` 最終結果為 **20/20 PASS**。Frame 1～20 全部依 frozen manifest 順序完成 replay、strict Mesa ingest、PvrGPU execution 與 17-counter byte-for-byte exact compare；沒有使用 PNG 決定結果，也沒有跳過任何失敗 frame。

主要 artifact root：

```text
$PVRGPU_WORK_ROOT/out/rdc-counter-batch/20260830T035010Z-70441
```

該 root 的 `run.txt` 記錄：

```text
dut_mode=renderdoc-mesa-poc-runner
png_compare=false
counter_compare=17-fields-exact
status=PASS
frames=20
```

| Frame | Workload family | 狀態 |
|---:|---|---|
| 1～4 | fill/depth/blend | PASS |
| 5～7 | triangle setup/cull | PASS |
| 8～11 | single/multi-attribute | PASS |
| 12～15 | 1/2/4/8 varyings | PASS |
| 16～20 | nearest/bilinear/trilinear texture | PASS |

每個 `NNN-case/` 都有 `counter_golden.txt`、`counter_pvrgpu.txt` 與 `result.txt`；20 個 `result.txt` 全為 `status=PASS`。正式 run 沒有產生 `counter_diff.txt`。

舊的 builtin adapter baseline 仍保留為有用的非正式回歸：Frame 1～20 為 20/20 exact match，artifact roots 是 `20260830T020445Z-34199` 與 `20260830T020536Z-34494`。它不計入上述正式 Mesa-ingest 進度。

## Debug 規則

發生 mismatch 時只做以下循環：

1. 先確認失敗階段：replay、capsule translation、PvrGPU ingest/protocol，或 counter compare。
2. 若已進 compare，看 `counter_diff.txt` 的第一個不同 counter；同時對照 Golden report、`pvrgpu/stdout.jsonl` 與 `pvrgpu/stderr.log`。
3. 若尚未進 compare，先看 `pvrgpu/mesa-poc/player.stderr.log`、`translator.stderr.log` 與 `command.txt` 是否存在。
4. 修正 Mesa/POC bridge 或 PvrGPU source code，重建後只重跑失敗 case。
5. 此 case exact match 後，再從下一個 frame 繼續 batch。

不因 PNG 相同而判 PASS，也不在失敗尚未修好時跳到下一個 `.rdc`。
