# Manhattan RDC Debug 方法與流程

本文件供 agent 重複使用：以 `gl_manhattan31_capture_1.rdc` 為例，從完整
capture 縮小到第一個有差異的 DrawList，使用 llvmpipe 作為獨立參考，沿著
Gallium → PCO → SystemC → framebuffer readback 找出差異來源。
本文只描述方法，不記錄個別除錯進度、問題清單或測試結果。

## 1. 基本原則

- 先讀 repository 的 `README.md` Ground Rule、driver README 與
  `docs/PVRGPU_DRIVER_COMMAND.md`，遵守「No Prepared Answers」。
- PvrGPU 的 geometry、shader、pixels、counters 必須由真實輸入經原生
  driver／模型產生。llvmpipe 的輸出只供比較，不得回填到 PvrGPU。
- 不用 capture 名稱、shader hash、預期 draw 數或 golden pixels 決定執行結果，
  也不跳過造成差異的 draw。hash 用於識別輸入與版本，不用於選擇渲染行為。
- 每次先定位第一個差異，再修改負責該語意的層；不要用放寬容差掩蓋上游錯誤。
- 原始碼與文件留在 repository；Mesa checkout、build、capture、log、PNG、raw
  attachments 與臨時工具放在 iCloud repository 以外。
- 先看 `git status`，保留既有修改。多 agent 必須明確分配檔案所有權，
  協調共用 build／install，不能一邊執行重播一邊覆寫它正在載入的 library。

## 2. 固定輸入與執行環境

從 repository root 載入本機設定；以下路徑是本機範例，換機須重新確認：

```bash
PVRGPU_REPO='/Users/linwanyi/Library/Mobile Documents/com~apple~CloudDocs/Codex/PvrGPU'
cd "$PVRGPU_REPO"
set -a
source config/local.env
set +a

PVRGPU_CAPTURE='/Users/linwanyi/Downloads/_Codex/GPU_TestPatterns/4.gl_31_manhattan/recorder/trace/gl_manhattan31_capture_1.rdc'
PVRGPU_RENDERDOC_ROOT='/Users/linwanyi/Downloads/_Codex/Working/build/renderdoc-mesa'
mkdir -p "$PVRGPU_OUTPUT_ROOT"
PVRGPU_DEBUG_RUN="$(mktemp -d "$PVRGPU_OUTPUT_ROOT/manhattan-debug.XXXXXX")"

git status --short
git rev-parse HEAD
shasum -a 256 "$PVRGPU_CAPTURE"
```

為每次執行保存一份 receipt，至少包含：

- capture 絕對路徑與 SHA-256。
- repository commit，以及本次實際使用的未提交修改／patch。
- Mesa source、build、install prefix；runner、RenderDoc player 與 SystemC bridge 路徑。
- 實際 `libgallium`、bridge、player 的 SHA-256 與相容的 API 版本。
- 完整命令、影響 backend 的環境變數、timeout 與輸出目錄。

PvrGPU runner／bridge 與 Mesa Gallium driver 是不同 build 產物，不能只重建其中
一邊就假定全部更新。確認 Mesa build 編譯的是 repository driver 還是另一份副本；
以 `compile_commands.json`、build 設定與檔案比較確認，不能只看路徑名稱。
外部 Mesa patch 的套用方式參考 `docs/PCO_LOWERING.md`。

除錯優先使用獨立 Mesa prefix 與固定 bridge 副本。`--allow-stale-mesa` 只能用於
刻意重現舊環境，不能作為新原始碼已被執行的證據。

## 3. 先取得完整重現，再建立 llvmpipe 參考

### 3.1 單一 pattern 重現

```bash
bash "$PVRGPU_REPO/script/run_regression.sh" \
  --filter '(^|/)gl_manhattan31_capture_1\.rdc$' \
  --out-dir "$PVRGPU_DEBUG_RUN/regression"
```

若需要明確指定獨立 Mesa prefix，直接使用 native runner：

```bash
"$PVRGPU_BUILD_DIR/bin/pvrgpu" "$PVRGPU_CAPTURE" \
  --mesa-prefix "$PVRGPU_MESA_PVRGPU_PREFIX" \
  --outdir "$PVRGPU_DEBUG_RUN/native-full"
```

保存 stdout、stderr、runner report、driver counter、driver command、model JSONL
與附件。先分清停止在哪一階段：capture 載入／probe、shader 編譯、command
transport、模型執行，或 framebuffer 讀回；exit code 本身不足以定位原因。

### 3.2 建立 action 與 Gallium submission 對照

使用有 counters 支援的 llvmpipe Mesa。以下以專案使用的 RenderDoc helper 產生
action 清單、Gallium DrawLists 與 shader 資料；先確認 helper 路徑存在：

```bash
(
  mkdir -p "$PVRGPU_DEBUG_RUN/reference/phase1/shaders"
  unset PVRGPU_SYSTEMC_API_LIB PVRGPU_SYSTEMC_BRIDGE
  unset PVRGPU_SYSTEMC_JSONL_OUT PVRGPU_DRIVER_COMMAND_OUT
  unset PVRGPU_DRIVER_COUNTER_OUT PVRGPU_RDC_CASE_NAME PVRGPU_RDC_TRACE_DRAW_ACTIONS
  unset MESA_COUNTER_FRAME_TIME_MS MESA_COUNTER_FORCE_END

  export EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1
  export GALLIUM_DRIVER=llvmpipe MESA_LOADER_DRIVER_OVERRIDE=swrast
  export MESA_SHADER_CACHE_DISABLE=true
  export DYLD_LIBRARY_PATH="$PVRGPU_LLVMPIPE_MESA_PREFIX/lib"
  export LIBGL_DRIVERS_PATH="$PVRGPU_LLVMPIPE_MESA_PREFIX/lib/dri"
  export RENDERDOC_MESA_EGL_PATH="$PVRGPU_LLVMPIPE_MESA_PREFIX/lib/libEGL.dylib"
  export RENDERDOC_MESA_GLES_PATH="$PVRGPU_LLVMPIPE_MESA_PREFIX/lib/libGLESv2.dylib"
  export MESA_COUNTER_REPORT_PATH="$PVRGPU_DEBUG_RUN/reference/phase1/Mesa-Raw.md"
  export MESA_COUNTER_DRAWLIST_PATH="$PVRGPU_DEBUG_RUN/reference/phase1/GalliumDrawLists.csv"
  export MESA_COUNTER_SHADER_PATH="$PVRGPU_DEBUG_RUN/reference/phase1/shaders"
  export MESA_COUNTER_FRAME_SELECTION_MS='ordered actions; marker = DrawList + 1'

  "$PVRGPU_RENDERDOC_ROOT/bin/renderdoc-rdc-splitter" \
    --phase1 "$PVRGPU_CAPTURE" \
    "$PVRGPU_DEBUG_RUN/reference/phase1/actions.tsv"
)
```

macOS prefix 若沒有 `lib/dri/swrast_dri.dylib`，在該次獨立輸出目錄建立指向
已確認 `libgallium` 的 DRI staging link，再設定 `LIBGL_DRIVERS_PATH`；
不要無意間使用另一份 Mesa。不同 helper 的介面以其原始碼／usage 為準。

完整 reference 還應保存最後 framebuffer、各階段 attachment 資訊與 counters，
並確認 capture、解析度和附件選擇與 PvrGPU 相同。

### 3.3 正式 full-frame player 與最終附件 receipt

正式路徑使用 repository 的 `tools/renderdoc-mesa-player.cpp`，不要把分段
診斷 helper 的中途 PNG 當成完整 replay 輸出。私有建置 script 只編譯這個
helper、PNG writer 與 audit parser，不會重建 Mesa／SystemC，也不修改已安裝的 RenderDoc
或 `config/local.env`。指定與 header 相容、支援公開 `ReplayEventRange` API
的標準 RenderDoc library；單次完整範圍不需要私有的跨範圍 scope 修正。

```bash
export PVRGPU_RDC_PLAYER="$(
  CXX=/usr/local/opt/llvm/bin/clang++ \
  bash "$PVRGPU_REPO/script/build_renderdoc_player.sh" \
    "$PVRGPU_RENDERDOC_ROOT/src" \
    "$PVRGPU_RENDERDOC_ROOT/build-homebrew-x86_64/lib/librenderdoc.dylib"
)"

python3 "$PVRGPU_REPO/tests/check_renderdoc_player_guards.py" \
  "$PVRGPU_RDC_PLAYER" "$PVRGPU_CAPTURE"

# Scope unit tests use a fresh private build. Optional library arguments add
# a real Mesa error-drain and callback/filter restoration probe.
CXX=/usr/local/opt/llvm/bin/clang++ PVRGPU_TEST_SANITIZERS=1 \
  bash "$PVRGPU_REPO/script/run_rdc_gl_debug_scope_unit.sh"
EGL_PLATFORM=surfaceless GALLIUM_DRIVER=llvmpipe \
  MESA_LOADER_DRIVER_OVERRIDE=swrast \
  LIBGL_DRIVERS_PATH="$PVRGPU_LLVMPIPE_MESA_PREFIX/lib/dri" \
  DYLD_LIBRARY_PATH="$PVRGPU_LLVMPIPE_MESA_PREFIX/lib" \
  CXX=/usr/local/opt/llvm/bin/clang++ \
  bash "$PVRGPU_REPO/script/run_rdc_gl_debug_scope_unit.sh" \
    "$PVRGPU_LLVMPIPE_MESA_PREFIX/lib/libEGL.dylib" \
    "$PVRGPU_LLVMPIPE_MESA_PREFIX/lib/libGLESv2.dylib"

"$PVRGPU_BUILD_DIR/bin/pvrgpu" "$PVRGPU_CAPTURE" \
  --player "$PVRGPU_RDC_PLAYER" \
  --mesa-prefix "$PVRGPU_MESA_PVRGPU_PREFIX" \
  --outdir "$PVRGPU_DEBUG_RUN/native-formal"
```

編譯器、RenderDoc library 和 prefix 路徑須依本機調整；重播前仍須把
`PVRGPU_SYSTEMC_API_LIB` 指向已確認 API 相容的固定 bridge 副本。runner
支援 `PVRGPU_RDC_PLAYER`／`--player`；沒有顯式指定時使用
`$PVRGPU_BUILD_DIR/bin/pvrgpu-rdc-player`，不能默默退回外部舊 helper。
llvmpipe reference 也可用相同 helper 配合 llvmpipe prefix 建立完整輸出。

正式 helper 的執行契約：

1. `OpenCapture` 與 end-state metadata preview 期間關閉 native API／bridge、
   command、model JSONL、driver counter 與測量 marker。preview 只保留最後
   pipeline 的 attachment resource、mip、layer、格式與 live GL 物件身分，
   不讀取任何初始化圖片作為答案。
2. 用 `SetFrameEvent(0, true)` 恢復 capture 的 initial contents／初始狀態，
   期間保留獨立同步 callback observer，完成 `glFinish` 並驗證還原過程無錯誤，
   不把 initial-copy error 當 preview noise 清掉。native backend 將此段 driver
   events 分開寫到 receipt 路徑加 `.initial-copy-driver-counter.txt`；要求真
   flush evidence，只接受已核對的 driver state／raw-copy event allowlist，並
   檢查 status／failure flags；未知事件、截斷、error／fail／unsupported／declined
   或需要 shader/native submission 的 copy 路徑都拒絕。只接受原始資源 transfer，不假裝在隔離期間執行
   shader。成功後才恢復 native 環境。frame 內的 resource dependency、graphics、compute
   與 transfer 都留在同一個正式 ordered replay；不把中途 warmup 關閉 native。
3. 一次 `ReplayEventRange(1, lastEvent)` 完成整個 frame，不在 query／TF scope
   中途切段。action count 只是報告 metadata，不設定預期 draw-count flush gate。
   有 slice／split metadata 的 capture 明確拒絕，不能假定省略的 dependencies。
4. 在 native range 前，於真正 replay EGL context 安裝同步 GL debug callback，
   在獨立 debug group 啟用所有訊息。保留並 chain 原 callback／userdata，記住
   原 enable bits；group 的 filter inheritance／pop 負責精確還原原本各
   source、type、severity、ID filters。記錄所有 ERROR type 或 HIGH severity，
   不只檢查結束時的 `glGetError`／RenderDoc messages：replay 內部可能已取走
   sticky error，仍必須拒絕成功結果。先檢查 captured calls，會替換 callback、
   改 filter、關閉同步 observation、切換 context，或 pop 超出自身 debug-group
   範圍的 capture 明確拒絕；不靠最後才重新開啟 callback 來假裝全程有監測。
5. 在同一 replay context 呼叫所選 Mesa 的 `glFinish`，保留 native
   counter／report scope，檢查 GL error、retained callback errors、RenderDoc
   fatal status 與 validation messages；驗證 callback、enable bits、group depth
   未被替換。再核對 completed draw FBO 的物件、subresource、extent，直接用
   `glReadPixels` 讀回；保存／還原 read FBO、read buffer、PBO 與 PACK state。
   不以 `SaveTexture` debug draw 或另一次 replay 產生最後圖片。
6. 成功讀回並驗證／還原原 callback、debug group/filter 與 enable state 後，
   才關閉測量 scope、atomic publish PNG 與 `pvrgpu.rdc-final-output.v2` receipt。
   receipt 必須含 `api_error_capture=synchronous-gl-debug-callback` 與
   `debug_callback_verified=true`、`initial_contents_restored=true`；native receipt
   另外指定 `initial_copy_driver_counter_path`。不接受舊 v1 或僅末尾
   `glGetError` 的證明。
   最後附件只支援單 sample、非 layered
   texture／renderbuffer；integer PNG conversion 或未支援 readback 明確拒絕。

直接呼叫 helper 的介面是 `TRACE.rdc OUTPUT.png [OUTPUT_trace.md]`。
PvrGPU 必須提供 `PVRGPU_RDC_FINAL_OUTPUT_RECEIPT`、native API library、driver
counter 與 model JSONL 路徑；正式 runner 負責配置這些新輸出路徑。helper
拒絕既存 final PNG／receipt／可選 trace，並解析既存 parent symlink 後確認
所有 input/output 路徑不同，避免覆寫 capture 或把舊 artifact 當成本次結果。

receipt 不是 native correctness 的單獨證明。runner 還必須 canonical match
RDC path、核對完整 replay／finish／API error 欄位、保存 RDC／PNG hash，驗證
initial-copy audit 路徑與無錯誤的 flush（不加進 native counters），再驗證
所有 graphics model reports 和 compute submit／done 完整配對，拒絕任何
driver／model error 或截斷。計數只累加真正完成的 report，最終圖片必須來自
receipt 指定的 `completed-replay-attachment`，不能用其中某次 model PNG
替代。比較結論仍須另外核對 llvmpipe 的實際 pixels／counters。

## 4. 正確定義「每個 DrawList」

三種編號必須分開：

| 名稱 | 定義與用途 |
| --- | --- |
| DrawList index | 本流程使用零起算的 RenderDoc leaf draw action 編號 |
| Event ID | capture 中實際事件邊界，不等於 DrawList index |
| Gallium submission | driver 實際收到的 draw，可能包含額外 helper work |

每筆對照至少保存 `draw_index`、`start_event`、`end_event`、action 的 index／vertex
數、instance 數，以及對應的 Gallium submission ID、shader 與 framebuffer。

不能假設一個 action 恰好等於一筆 Gallium draw。以 marker、IA 數量、instance、
shader 綁定與事件順序交叉確認；有多個候選就保留「映射未解決」，不要猜一筆。

計數分成三類保存，不混成目標 draw 的數值：

1. application draw 本身的工作。
2. 同一事件區間中的 mipmap、blit、其他 helper 工作。
3. draw 之間的 compute／其他非 draw 工作。

零 indices／零 instances、沒有 query、資料缺失與「實際量到零」各自表示。
沒有量到的欄位用 `null`／未量測，不用補零湊出相同的結果。

## 5. Ordered-prefix replay：由第一筆逐步向後

### 5.1 重播契約

對 DrawList N，從相同 capture 初始狀態依序執行到它的 Event ID，而不是只呼叫
第 N 個 draw。必須保留前面的 buffer／texture 更新、clear、mipmap、blit、compute、
barrier、framebuffer 切換及其他資源依賴。

一次連續 prefix session 的概念流程如下：

```text
讀取 capture / 列舉 actions
→ 還原 capture 初始資源與狀態
→ 完整執行上一邊界到本 draw 的事件區間
→ 在 replay GL context 同步，確認原生模型完成
→ 保存本次 submission 的 counters 與所有 attachments
→ 繼續下一區間
```

不同 prefix run 都重新由初始狀態開始。同一次 session 內則連續向前，不能每個
draw 都清空 attachments，也不能在 snapshot 時重新 replay，卻把重複工作算成新 draw。

切段前須檢查 replay API 的副作用：query、conditional rendering 與 Transform Feedback
的 Begin／End scope 可能跨越 draw 邊界。連續 range 不能在每段末端自動結束它們，
也不能用忽略原始 End 或清掉 GL error 代替正確 scope。另行驗證 scope 跨段存續、
重設 frame 時的清理，以及非零 Transform Feedback／query 結果。將 replay 原已存在的
GL error 與 snapshot 自己造成的 error 分開保存；snapshot 成功不代表 replay 沒有錯誤。

capture 初始化可能掃描甚至執行整個場景，所以「指定 DrawList N」不表示載入階段
只會碰到 N 之前的 shader。journal 必須分開記錄初始化、正式事件區間、同步和快照。

若使用只做載入／編譯的初始化模式，必須明確記錄該模式：保存並暫時移除會啟動
原生提交的 command／API／JSONL 環境設定；完成初始內容還原後，才恢復原設定。
正式 prefix 的依賴與 draw 必須全部啟用 PvrGPU／SystemC，不得以停用 backend
省略目標工作。另用完整執行模式檢查結果不依賴初始化捷徑。

trace action 數量只作為外部報告 metadata。不要把它當 native draw 上限、
完成條件或停止執行的開關；零工作 action 與 helper 會破壞這種假設。

### 5.2 Bounded helper 命令

本機 bounded helper 的工具目錄範例：

```bash
PVRGPU_DEBUG_TOOLS='/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/out/runs/manhattan-llvmpipe.L3027L'
```

該目錄中的 `run_bounded.py`、`bounded_player.cpp` 與另行固定的 helper binary 是外部
診斷工具，不是 repository 的正式 build target。執行前檢查原始碼、binary receipt、所依賴的
BenchScope／RenderDoc 路徑與 CLI；不要因為目錄存在就假定工具相容。搬機或臨時目錄
遺失時，依本節契約重建工具；可先搜尋 `run_bounded.py`，不要自動挑一份舊 binary。

```bash
rg --files "$PVRGPU_OUTPUT_ROOT" -g run_bounded.py
python3 "$PVRGPU_DEBUG_TOOLS/run_bounded.py" --help
```

從同一份已驗證的 build receipt 指定 helper、獨立 evidence auditor，以及 helper
實際連結的 continuous-range RenderDoc library；不要挑選工具目錄中同名的舊 binary。
以下是需要替換的路徑，不是自動偵測設定：

```bash
PVRGPU_BOUNDED_PLAYER='/absolute/verified-build/bounded-player'
PVRGPU_BOUNDED_AUDIT='/absolute/verified-build/bounded-audit'
PVRGPU_BOUNDED_RENDERDOC='/absolute/verified-renderdoc/lib/librenderdoc-range.dylib'
test -x "$PVRGPU_BOUNDED_PLAYER"
test -x "$PVRGPU_BOUNDED_AUDIT"
test -f "$PVRGPU_BOUNDED_RENDERDOC"
```

設定要檢查的末端 draw，先產生同範圍 reference，再執行 PvrGPU：

```bash
PVRGPU_LAST_DRAW=0

python3 "$PVRGPU_DEBUG_TOOLS/run_bounded.py" "$PVRGPU_CAPTURE" \
  --backend llvmpipe \
  --mesa-prefix "$PVRGPU_LLVMPIPE_MESA_PREFIX" \
  --player "$PVRGPU_BOUNDED_PLAYER" --audit-tool "$PVRGPU_BOUNDED_AUDIT" \
  --renderdoc-lib "$PVRGPU_BOUNDED_RENDERDOC" \
  --draw "$PVRGPU_LAST_DRAW" --every-draw \
  --outdir "$PVRGPU_DEBUG_RUN/reference-prefix" --timeout 300

python3 "$PVRGPU_DEBUG_TOOLS/run_bounded.py" "$PVRGPU_CAPTURE" \
  --backend pvrgpu \
  --mesa-prefix "$PVRGPU_MESA_PVRGPU_PREFIX" \
  --bridge "$PVRGPU_SYSTEMC_API_LIB" \
  --player "$PVRGPU_BOUNDED_PLAYER" --audit-tool "$PVRGPU_BOUNDED_AUDIT" \
  --renderdoc-lib "$PVRGPU_BOUNDED_RENDERDOC" \
  --draw "$PVRGPU_LAST_DRAW" --every-draw \
  --reference-dir "$PVRGPU_DEBUG_RUN/reference-prefix" \
  --outdir "$PVRGPU_DEBUG_RUN/native-prefix" --timeout 300
```

`--draw` 為零起算，`--event` 可改用實際 Event ID，兩者擇一。
若最後一筆 draw 後還有 transfer、barrier 或其他事件，完整 prefix 應以 capture 的
最後 Event ID 為界，明確執行與稽核尾段，不能只停在最後 draw。
`--every-draw` 保存 prefix 中每筆 draw；不加時只保存所選末端的附件。
`--compile-only-init` 是前節所述的顯式初始化模式，只在理解且記錄其邊界後使用。
`--outdir` 必須尚不存在；每次改末端 draw 或 runtime，使用新的目錄。
timeout 按實際模型成本設定，timeout 只代表沒有完成，不代表像素錯誤。
外層 runner 必須獨立稽核逐區間的 submit／done／新 model reports 與 snapshot scope，
拒絕截斷 journal、失敗後繼續執行，以及 child exit 0 卻缺少完成證據的結果。
診斷工具成功完成仍不是正式 full-frame 或圖片正確性的 PASS。

可先用較大的 prefix 找到差異區間，再二分末端 Event ID；每次二分仍完整執行
所需前綴。定位後，回到逐 draw 比對，檢查差異第一次出現的位置。

## 6. 同步、readback 與比較

### 6.1 證據必須來自同一次提交

在 replay GL context 呼叫實際 Mesa `glFinish`，再進入可能切換 debug context 的
texture snapshot API。但「呼叫已返回」不等於模型已完成：還須確認 native submission、
completion、generation／序號與 framebuffer readback。

每個事件區間記錄 driver log 的起訖 byte offset，並立即另存模型 JSONL。
共用 `model.jsonl` 可能被覆寫，也可能持續 append；先確認實際行為，不要拿最後一份
檔案或固定第一筆／最後一筆 counter 代表整個 prefix。記錄提交前後的檔案身分與 byte
offset，辨識本區間新增的 reports；若一個區間含多次 flush，保留每次提交並按 counter
語義彙整。排除舊檔、其他 helper／clear 或失敗後的舊快取。

在 replay range 返回及同步完成後、保存 snapshot 前，分別檢查該區間的 compiler、
driver、model 錯誤。`unsupported_draw`、編譯失敗或 CPU present fallback 必須令
本區間失敗並停止向後重播，不能因 `glFinish` 返回、process exit 0 或最後有 PNG
就略過。非零 draw 沒有本區間的 native completion，應標記證據缺失／待查；真正
零工作 action 則獨立分類，不能借用上一筆 model report 冒充這一筆。

### 6.2 附件比較

- 保存每個實際 color target，以及存在的 depth／stencil，不能只檢查 color0。
- 固定並記錄尺寸、格式、mip、layer／slice、sample、方向、row pitch 與 raw byte 數。
  `mip 0 / slice 0 / sample 0` 只是明確的選擇，不代表已覆蓋全部子資源。
- action 的 resource ID 不一定包含實際 attachment mip／layer。range replay 若不更新
  RenderDoc pipeline-state cache，須在仍 current 的 replay GL context 查詢實際
  framebuffer attachment，再按該 subresource 保存。不要為了刷新 cache 呼叫會重新
  replay 的 API；也不要把整張 texture 的 base-level 尺寸當作非零 mip 的尺寸。
- MSAA 的 sample readback 與 resolve 分開；不要把 resolve 隱藏在比較裡。
- raw bytes 作為主要資料；PNG 用於目視定位。深度 PNG 若需額外 debug shader，
  優先保存 raw depth，將視覺化工作與 application draw 分開。
- 原生路徑優先從已完成的 replay framebuffer 直接 `glReadPixels`，保存並還原
  read FBO、read buffer、PBO 與 PACK state。RenderDoc 的 `SaveTexture`／
  `GetTextureData` 可能插入額外 shader draw；此類工具工作與錯誤必須有獨立
  scope，不可混入 application counters，亦不可用 CPU presentation 代替原生
  shader。若 readback 轉成 RGBA8，分開記錄原 attachment 格式與保存格式。
- readback 回傳非空 buffer 仍不足以證明資料有效。depth／stencil 須確認 API／extension
  支援與 GL error；必要時在實際 replay context，以已驗證的 `glReadPixels` 格式／型別
  交叉檢查。保存並還原 read framebuffer、pixel-pack buffer／布局等狀態，且確認讀的是
  所選 action 的附件。GL 的 `UNSIGNED_INT` depth 與 driver 的 packed Z24 並非相同
  bit layout，轉換須依格式定義，不可直接當作同一種 raw bytes。
- 先比格式與 byte count，再比 hash／內容。需要量化時記錄差異 pixel 數、
  每通道最大／平均誤差及差異區域。
- 格式 padding 與真正的 depth／stencil bits 分開解讀；浮點差異依規格與運算精度
  說明容差，不能為了通過單一 capture 調整門檻。
- packed normalized target 須區分模型的 canonical attachment、driver 打包後的
  實體格式，以及 PNG 展開。RGB565 的 5／6-bit 色階以 bit replication 展開，
  不可用另一種四捨五入轉換製造比較差異；但 float → RGBA8 → RGB565 與直接
  float → RGB565 的兩段量化差異屬於真正的附件值差異，不能當成純顯示誤差。
  對應 memory counters 也須標示模型實際使用的 bytes/pixel，不能把 canonical
  RGBA8 流量宣稱為原生 RGB565 流量。
- 無可見片元的 draw 可能正確地不改變影像，但仍要檢查 IA、VS、clipping／culling
  與後續附件內容；兩張零圖相同不能單獨證明有執行 draw。

### 6.3 Counter 比較順序

先比同一 action 對應的工作，不先比整張 frame 的總和：

1. IA vertices／primitives、indices、instances。
2. VS invocations 與 vertex reuse。
3. clipping／culling、setup primitives。
4. fragment invocations、discard、depth／stencil。
5. shader ALU／texture／memory 與 texel fetches。
6. 另外歸屬 compute、helper 及其他事件區間的工作。

先確認 counter 定義是否相同，特別是 helper lanes、頂點重用、取樣 tap 與 cache。
架構相關的計數差異先解釋語意，不直接要求每個欄位相等；未校準的 SystemC cycles
也不能直接當作 llvmpipe 的 wall-clock 效能。

## 7. 從第一個差異往下追

按資料流檢查，優先找最早失真的層：

| 層 | 要保存／比對的資料 |
| --- | --- |
| GL／Gallium | VBO、EBO、UBO、texture、sampler、viewport、MRT、depth/stencil、blend、instance state |
| NIR／PCO compiler | 原始與 lowering 後 NIR、完整 PCO binary、stage ABI、shared／descriptor layout |
| Decoder／ISS | 指令 offset、register bank／index、modifier、讀寫範圍、分支、SMP／LD／WDF continuation |
| 幾何與 fragment pipeline | vertex outputs、clip／window coordinates、varyings、quad／helper 身分、depth 與 discard |
| Memory／readback | 附件所有權、LOAD／CLEAR、alias、format／stride、generation、DRAM 原始輸出 |

完整 PCO bytes 可透過環境設定保存：

```bash
PVRGPU_PCO_DECODE_DUMP=1 \
  python3 "$PVRGPU_DEBUG_TOOLS/run_bounded.py" "$PVRGPU_CAPTURE" \
    --backend pvrgpu \
    --mesa-prefix "$PVRGPU_MESA_PVRGPU_PREFIX" \
    --bridge "$PVRGPU_SYSTEMC_API_LIB" \
    --player "$PVRGPU_BOUNDED_PLAYER" --audit-tool "$PVRGPU_BOUNDED_AUDIT" \
    --renderdoc-lib "$PVRGPU_BOUNDED_RENDERDOC" \
    --draw "$PVRGPU_LAST_DRAW" \
    --outdir "$PVRGPU_DEBUG_RUN/native-pco-dump" --timeout 300
```

保存 stderr 中的 `PCO_DECODE_BINARY`。driver 的短版 hex 摘要可能沒有涵蓋失敗 offset，不能據此
猜指令。使用該次 runtime 的 decoder／符號，追到 Mesa 公開編碼與真實 NIR 語意；
不要改成辨識 binary 後回傳手寫 shader。

每次只改一類語意，新增不依賴 capture 名稱的最小測試：改變 register bank、
binding 順序、尺寸、索引與邊界條件，確認修正不是針對一組固定資料。

### 7.1 非有限值與 shader 中間值

遇到 NaN／Inf 時，從失敗的 sampler／PIXOUT 往前找第一個產生它的指令，記錄
operand raw bits、register bank、實際像素、primitive／quad 與 helper 身分。
同時讀取 capture 的原始 attribute bytes，並在 llvmpipe 匯出同一位置的 shader
中間值；不能只因出現 NaN 就推定 PCO 或插值出錯。

若用 shader replacement 匯出 float bits，保留原 precision 與運算，只在目標事件
啟用診斷。相同 shader 可能被前面的 draw 共用，須確認診斷停用時的 prefix MRT
與未替換控制組逐 byte 相同。不要把修改過的 llvmpipe shader 作為原生輸入。

先區分規格明確定義的運算、應保留的 NaN，以及未定義輸入的實作政策。
地址／格式轉換邊界的安全處理不能回頭改寫 ISS operand，也不能跳過 shader 或
記憶體工作；測試須包含正常有限值控制組、非有限值、混合 helper quad、實際
texel reads，以及仍應拒絕的非法 descriptor／allocation。未定義像素與 llvmpipe
相同只能作為觀察，不能宣稱是規格要求。

### 7.2 保存原生 API 輸入以縮短重播

prefix 過長時，可在真實 native submit 邊界使用透明 shim 深拷貝單筆輸入。
保存完整 shader／ABI、vertex／index、shared／UBO、texture bytes、fixed-function
state、generation，以及該次原生 prefix 產生的所有 MRT／depth LOAD；短版 text
command 摘要不足以取代這些 payload。

shim 必須原封不動轉送實際提交；不支援的 archive shape 明確記錄「未捕捉」，
不得停止或替換原 draw。獨立 replayer 驗證 ABI、byte bounds、hash 與所有指標
重建，只允許重定位輸出路徑。更換模型 binary 時記錄新舊 hash，損壞的 archive
仍須拒絕。先用小型非零 MRT／depth fixture 確認捕捉前後輸出完全相同。

這是同一筆真實輸入的冷啟動重播，不是 cache／timing checkpoint；不得據此要求
cycle 相同。修正後仍須回到完整 ordered prefix，確認跨 draw 相依與正常執行流程。

### 7.3 正規 DrawList snapshot／resume

若要保存 DrawList N 完成後的整體狀態，讓新程序直接正常執行 N+1，使用
[DrawList Snapshot / Replay 方法](DRAWLIST_SNAPSHOT_REPLAY.md)。這與上一節的單筆
native API input capsule 不同：後續事件必須讀取接續執行所產生的新資源內容，不能逐筆
載入舊 capsule 覆寫前一步的結果。

### 7.4 UBO 越界與 compiler memory footprint

先把失敗的 native LD 記成 stage、block、GPU base、綁定 bytes、實際 address、
DWORD count 與指令 offset。以 `address - base` 計算相對範圍，核對 captured
buffer binding 的 offset／size、driver snapshot、shared descriptor 與原始 NIR
load；不能用 DRAM page 已配置或相鄰 buffer 有資料，取代 API 綁定範圍檢查。

若原始存取合法而 native load 越界，分別驗證兩個邊界：

1. 依 Mesa ISA 定義核對 PCO burst encoding、DWORD／byte 單位與 address
   arithmetic，不由失敗的 byte count 推測 decoder 應如何改。
2. 保存 vectorize、shrink、lowering 前後的 NIR。對每個 memory load 計算
   實際讀取區間，而非只看 shader 最後使用的分量；未使用但仍被 LD 讀取的
   lanes 也必須合法。合併／前端裁切／向量長度進位要一起檢查，並核對
   reswizzle、alignment、offset shift 與 range metadata。

建立不依賴 capture 的最小 pass pipeline，直接執行所選 Mesa source，避免
連結到舊 archive 的同名實作。以不同向量寬度、bit size、使用分量遮罩與 offset
驗證讀取範圍不擴張、使用值完全保留，並檢查 NIR validation／重複最佳化的
固定點。使用未修正 source 作負面對照，確認同一測試確實重現相同錯誤；再以
新 runtime 重播原 capture，不修改 UBO 大小、填補越界資料或替換 shader 答案。

### 7.5 分離取樣精度與反射放大

當 color／depth coverage 相同，但反射附件出現大幅差異時，先追蹤完整
2×2 fragment quad，不只追蹤一個最終像素。保存各 lane 的 UV、normal map
回應、未量化 normal、reflection vector、cube face、derivatives、LOD 與 mip
weight；encoded normal 相同不代表 shader 的中間 normal 完全相同。

分別重建 normal-map LOD 與 cube LOD。exact log2／fast approximation、先投影
再差分／投影的導數、coarse／per-lane footprint 都要分開，不把其中一種差異
套用到所有像素。用原始 texture bytes 與實際 filter／wrap／mip 狀態核對 sample。

需要因果控制時，只在獨立診斷 replay 中修改單一取樣操作，例如以 native trace
實際觀測到的 LOD 呼叫 `textureLod`；其他材質、shader 計算與 texture bytes
保持不變。保存修改前後 source 與 hash，核對上游中間值未改變，再以真正
GL readback 確認輸出變化。控制 shader 的結果不是原 capture 的 PASS 證據。
只解釋已驗證的像素／quad；殘差要保留，並查相應 API 規範是否容許該精度選擇，
不能為了模仿 llvmpipe 而改掉原本合法的模型算法。

實作上要明確分流，不可臨時改 source：Capture/Play 的 `draw-serialized`
比較設定 `PVRGPU_TEXTURE_LOD_MODE=llvmpipe`，保留 reference renderer 可觀測的
fast-log2 mip weight；dEQP、未來真實硬體與 `render-pass-tbdr` performance mode
設定 `PVRGPU_TEXTURE_LOD_MODE=exact`。這是 SystemC elaboration-time 模型選項，
同一 process 內不得中途切換；結果報告必須記載所用模式。

### 7.6 Packed render target 精度

先查當下 attachment 的實際 channel bit sizes。RGBA8 PNG／`GL_UNSIGNED_BYTE`
readback 無法證明 RGB10_A2 的低兩個 RGB bits 有被保留。使用對應的 packed
readback type，另以小型 highp constant shader 避開 LOD、mediump 與幾何誤差。

從格式定義選擇低位元、非對稱 RGB、alpha 邊界、半階上下值與 clamp 值，依序測：

1. shader → PBE → packed readback；
2. 精確 packed 資料 → initial LOAD → blend／channel mask → readback；
3. 同一像素連續片段寫入，驗證每個片段後量化，而非只在最後 pack 一次；
4. render → 下一個 draw 真正 `texelFetch` → 再寫回，驗證下游仍取得低位元；
5. clear、MRT、layer／sample 與相反通道排列，以及非法格式／payload 的拒絕。

同時核對 driver envelope、model format gate、LOAD、blend、DRAM、readback 與
texture descriptor；只修最後的 pack 不能恢復前面已丟失的 bits。以實際儲存
bytes 計算 memory traffic，顯示轉換不可回流到 native resources。
llvmpipe 的 store 也可能採用規範容許的截位，須以實際 source／最小 GL 控制與
規範分類，不把所有 1-code 差異稱為錯誤或無條件放寬整張圖的門檻。

## 8. Crash、heap corruption 與 sanitizer 流程

1. 找出真正終止的 child process，保存 player stderr、系統 crash report 與完整 stack。
   macOS crash reports 可從 `~/Library/Logs/DiagnosticReports` 找到。
2. 區分錯誤發生點與偵測點；allocator 報錯的 stack 不一定是越界寫入的位置。
3. 建立獨立的 sanitizer Mesa／compiler runtime，使用
   `-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined`。
   從 `compile_commands.json` 與 Ninja link command 取得真正編譯／連結參數，
   同時重建相依的 generated opcode headers／metadata，避免混用 ABI。
4. 先以最小 compiler fixture 重現，再跑保留依賴的 capture prefix。
   ASan／UBSan 通過不能取代像素與原生執行檢查。
5. RenderDoc interposition 若與 sanitizer 初始化衝突，使用先連結並初始化 ASan 的
   小型 launcher，進入 `main` 後才 `dlopen` 含 RenderDoc 的 player module；
   不把初始化衝突當作 driver 問題，也不關掉檢查器掩蓋它。
6. 使用相符的 dSYM／`llvm-symbolizer` 解析 library offset；以 LLDB 的
   `thread backtrace all`、`image list` 確認 stack 與實際載入的 libraries。

可重用的 compiler／transport 測試入口依修改範圍選擇：

```bash
bash script/run_mesa_pco_register_bank_unit.sh
bash script/run_mesa_pco_ubo_unit.sh
bash script/run_mesa_pco_texture_unit.sh
bash script/run_mesa_pco_texture_sequence_unit.sh
bash script/run_mesa_nir_shrink_load_unit.sh
bash script/run_mesa_resource_unit.sh flush
bash script/run_mesa_resource_unit.sh boundary
```

## 9. 修正後擴大範圍與回報方式

採用固定順序：最小單元測試 → 第一個差異 draw → 前後相鄰 draw → 更長 prefix →
完整 frame → 相關 dEQP／既有 graphics regression。每次 rebuild 後更新 receipt，
不要覆蓋前一次的證據。

完整 capture 的終點以最後 Event ID 為準；最後 DrawList 之後若仍有 blit、copy、
clear 或其他收尾事件，必須保留獨立 trailing range 執行與稽核，不能用「最後一筆
draw 已返回」代替整個 capture 完成。

模型側可依修改範圍重建並選取相關 CTest；先確認本機 CMake target 名稱：

```bash
cmake --build "$PVRGPU_BUILD_DIR" --target \
  pco-iss-test usc-cluster-texture-continuation-test \
  pco-initial-color-load-test pco-systemc-api-bridge-test -j4

ctest --test-dir "$PVRGPU_BUILD_DIR" \
  -R 'pco-iss|usc-cluster-texture|pco-initial-color|pco-systemc-api-bridge' \
  --output-on-failure
```

逐 DrawList 回報使用以下欄位，分開表達「有執行」與「比較相符」：

| 欄位 | 應記錄內容 |
| --- | --- |
| 範圍 | DrawList index、Event ID、對應 Gallium submission |
| 執行 | replay 返回、原生 submission／completion、readback 的各自證據 |
| Counters | PvrGPU／llvmpipe 的同語意數值；application、helper、compute 分列 |
| 附件 | 每個 target 的尺寸／格式／子資源與 raw comparison／誤差 |
| 第一個差異 | 最早不同的階段、相關 log／binary／raw 檔案路徑 |
| 下一步 | 往後擴大 prefix，或縮小該差異的最小重現 |

交接時提供可重跑的命令、環境 receipt、事件對照與證據位置；不要只提供一張 PNG
或一句 PASS。最終畫面必須來自完整所需事件經 PvrGPU／SystemC 的實際執行和讀回，
並明確區分「畫面已輸出」「原生工作完整」「與 reference 相符」三種結論。
