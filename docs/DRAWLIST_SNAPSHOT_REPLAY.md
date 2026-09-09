# DrawList Snapshot / Replay 方法

## 契約

這是完整 replay 功能狀態的 checkpoint，不是 PNG、單筆 native API input capsule，
也不是省略 shader 執行的 state-only prefix。

在 DrawList N 的完整 API chunk 執行完、同步及錯誤稽核通過後，保存當時的資源內容、
物件狀態與 replay 位置。新程序載入 checkpoint，再正常執行 N+1 以後的事件。
後續 draw 使用前一步實際產生的新資料；不可用舊的逐筆 input capsule 覆蓋它。

- DrawList 採零起算；`--through-draw N` 包含 N。
- 每次執行都產生一個新的 checkpoint，原 checkpoint 不改寫。
- 最後一個 DrawList 必須包含 capture 的 trailing events；terminal checkpoint 不能繼續往後。
- 還原的是 functional state。新程序的 model cache、時間與 counters 從冷啟動開始，
  不能把 suffix 的 cycle／cache traffic 當成原始完整 frame 的效能。
- llvmpipe 和 PvrGPU 各自保存自己的 checkpoint。不可將 llvmpipe 資料注入 PvrGPU 當成驗證。

## 執行組件

1. 具版本化 snapshot extension 的 RenderDoc library，提供 Save／Load ABI。
2. `tools/renderdoc-drawlist-player.cpp`：同步、完整區間執行、GL／native audit 與 snapshot 操作。
3. `script/run_drawlist_replay.py`：凍結 runtime identity、檢查 receipt，最後提交 manifest。

Player 必須直接連結指定的 snapshot RenderDoc library，不能把一般 RenderDoc library
當成已有 snapshot 功能，也不能只替換環境變數來切換另一份已連結的 library。

從相容的 RenderDoc source 與既有 build objects 建立私有 snapshot library：

```bash
bash script/build_renderdoc_snapshot.sh \
  "$RENDERDOC_SOURCE_ROOT" "$RENDERDOC_BUILD_BASE" "$NEW_SNAPSHOT_BUILD_DIR" --jobs 2
```

Build 會核對 repo 內的 source lock 與 patch，使用私有 source overlay／objects，
不修改原 source、既有 build 或已安裝的 library。輸出目錄必須尚不存在，並位於 repo、
source 與原 build 之外。保留 `build-receipt.json`、`source.patch`、`source-lock.json`
與 build logs，作為可重現的建置依據。

使用產生的 `lib/librenderdoc-drawlist-snapshot.dylib` 絕對路徑設定
`SNAPSHOT_RENDERDOC_LIBRARY`，再建立 player：

```bash
bash script/build_drawlist_player.sh \
  "$RENDERDOC_SOURCE_ROOT" \
  "$SNAPSHOT_RENDERDOC_LIBRARY"
```

此指令在新的私有目錄產生 player，輸出其絕對路徑，不改寫已安裝的 runtime。
執行時將該路徑設為下面的 `DRAWLIST_PLAYER`。Mesa prefix 和 bridge 也必須使用明確、
已凍結的實體 artifact；不要在 replay 執行中重新安裝或覆寫。

## 保存及接續

以下變數必須指向本次選定的真實路徑，輸出目錄的 parent 必須已存在：

```bash
CAPTURE='/Users/linwanyi/Downloads/_Codex/GPU_TestPatterns/4.gl_31_manhattan/recorder/trace/gl_manhattan31_capture_1.rdc'
RUN_ROOT='/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/out/runs'
```

第一次由 capture 初始狀態執行到 DrawList 147：

```bash
python3 script/run_drawlist_replay.py "$CAPTURE" \
  --player "$DRAWLIST_PLAYER" \
  --renderdoc-lib "$SNAPSHOT_RENDERDOC_LIBRARY" \
  --mesa-prefix "$MESA_PREFIX" \
  --backend pvrgpu --bridge "$NATIVE_BRIDGE" \
  --through-draw 147 \
  --outdir "$RUN_ROOT/checkpoint-147" \
  --timeout 3600
```

新程序載入 147，正常執行到 148 並再次保存：

```bash
python3 script/run_drawlist_replay.py "$CAPTURE" \
  --player "$DRAWLIST_PLAYER" \
  --renderdoc-lib "$SNAPSHOT_RENDERDOC_LIBRARY" \
  --mesa-prefix "$MESA_PREFIX" \
  --backend pvrgpu --bridge "$NATIVE_BRIDGE" \
  --resume "$RUN_ROOT/checkpoint-147/snapshot" \
  --through-draw 148 \
  --outdir "$RUN_ROOT/checkpoint-148" \
  --timeout 3600
```

同理，下一次以 `checkpoint-148/snapshot` 為來源執行到 149。
若使用 llvmpipe，改成 `--backend llvmpipe` 並移除 `--bridge`。

`--outdir` 必須尚不存在；重新測試要換新目錄。`--help` 不啟動 build 或 GL replay。

`--gles-version` 預設為 `3.1`；保存及接續必須使用相同、已實際驗證的版本。
不要僅依 capture header 強制提高版本：Car Chase 目前驗證的是 GLES3.1＋
所需 extensions，強制3.2會啟用 driver 尚未支援的 indexed blend-state 查詢。
此選項只設定 replay context 的版本，不會補上功能，也不能代替 RenderDoc
的 API alias 修正。Car Chase 另需相容的 D16 renderbuffer snapshot codec；
既有 V1 尚不支援，不能直接把上述 Manhattan checkpoint 流程視為已可用。

## Artifact 與相容性

每個成功輸出含：

- `snapshot/state.bin`：RenderDoc snapshot envelope 與各資源的完整 state blobs。
- `snapshot/manifest.json`：最後原子提交的完成標記；包含 capture、state、runtime SHA-256、
  ABI、DrawList／event boundary、parent checkpoint 與 engine receipt。
- `replay.json`：執行端的完成證據。
- `stdout.log`、`stderr.log`、`invocation.json`；native backend 另有本次區間的 driver／model evidence。

只有 `state.bin`，或只有 child exit code 0，都不能判定保存成功。
必須由 wrapper 驗證完整 receipt、檔案 identity 與邊界，再提交 manifest。
中止或錯誤的目錄可保留作為診斷資料，但不能當成可接續的 checkpoint。

Capture 以內容 SHA-256 識別；相同內容改檔名不影響相容性。SHA 不得用來選取 shader 行為。
Snapshot 目錄可以搬移，manifest 中的舊絕對路徑只作 provenance。

預設要求相同 backend、player、RenderDoc、Mesa／EGL／GLES 和 bridge binary。
Mesa 的 DRI loader 與實際 libgallium 實作分別記錄，不得只核對 loader shim。
Runtime identity v2 另外固定 GLES override；改版本必須重建 checkpoint，
`--allow-model-change` 不放寬這項限制。舊 runtime identity v1 的版本固定為 3.1，
只可在 3.1 下接續，不能把舊 manifest 補上 3.2 就當成相容。
只有修正 model bridge 時，才可明確使用 `--allow-model-change`；它只允許 bridge 改變，
不放寬其他 runtime identity。使用者必須確認保存的 prefix 功能結果在修正後仍然有效。
若修正會影響 prefix，必須從 capture 初始狀態重新建立 checkpoint。

## 保存邊界及限制

只能在完整 API chunk 之後保存，不能從 multi-draw chunk 中間接續。
保存前必須確認沒有未完成工作或錯誤，不能以 `glFinish` 返回或 pending count 為零代替稽核。

V1 使用初始內容 codec 還原 supported capture-live objects，並檢查新 controller 的資源
allocation／program identity 能與 checkpoint 相容。無法完整重建的情況必須拒絕，不能
忽略未知物件、遺失 alias 關係，或用零填充補齊。

Texture 必須保存實際配置的所有 mip／layer，不能以當下的 `GL_MAX_LEVEL` 裁掉資料；
suffix 可能再次開放被隱藏的 mip。保存後重新讀回確認原狀態未被修改，載入後重新讀回
所有資源及 context 確認與 archive 相符，才可回報成功。UI pipeline metadata 採延遲查詢，
不能在成功驗證後又以查詢函式改變 GL binding。

活動中的 query、conditional rendering、transform feedback、跨界 debug groups、
mapped／imported storage、未支援的 MSAA／texture view／renderbuffer，以及無法還原的
query／TF／sync result 相依關係，均應先得到明確支援，否則不得建立可用 checkpoint。
若某個 DrawList 位於不支援的作用域內，選擇同一程序中該作用域完整結束後的邊界。

OpenCapture 的初始化工作與真正的 native interval 分開。Resume 不得呼叫
`SetFrameEvent(saved_event)` 或執行 prefix 來猜回 state；必須載入 archive 後直接執行 suffix。
資源保存／還原只能使用 lossless raw transfer；不能偷偷執行 helper shader 或 CPU present
來生成應由實際 draw 計算的內容。

## 等價驗證流程

新增格式、狀態或修正 restore code 時，至少驗證：

1. 同一 capture、同一 runtime：不中斷執行到 M，保存對照。
2. 執行到 N 保存，結束程序；新程序載入 N，再正常執行到 M。
3. 比較所有相關 texture mip／layer／sample、buffer bytes、附件、uniform／binding 狀態；
   格式與內容分開檢查，不能只看可視化 PNG。
4. 再從 M 接續下一筆，確認 M 新產生的結果確實向後傳播。
5. 檢查 resume native interval 沒有重播 prefix，保存／還原 scope 沒有 shader 計算或 error。
6. 做錯誤注入：截斷、改內容、換 capture／runtime、錯邊界、未支援狀態，均不得提交 manifest。

使用小型、可獨立計算預期值的相依 draw capture，先做逐 byte 比較，再驗證真實 pattern。
Model／driver 的功能差異與 checkpoint 等價性是兩個不同問題：checkpoint 不應替原來的
native 錯誤「修正」畫面，也不能把原來錯的 prefix 標成已經與 llvmpipe 一致。

`--verify-color` 可另外讀回已完成、當前綁定的 color0，供小型 fixture 獨立比對。
此驗證 codec 僅接受 single-sample 2D RGBA8，輸出 `color.rgba` 與 receipt metadata，
不做 helper shader conversion，也不代替完整 snapshot state 的驗證。

命令列及 pre-GL 拒絕測試：

```bash
python3 -m unittest discover -s tests -p test_drawlist_replay_cli.py
python3 tests/test_drawlist_player_guards.py "$DRAWLIST_PLAYER"
python3 -m unittest discover -s tests -p test_build_renderdoc_snapshot.py
python3 -m unittest discover -s tests -p test_inspect_drawlist_snapshot.py
```

可以用只讀 inspector 列出 archive 的資源 ID、格式 signature 與內容 digest：

```bash
python3 tools/inspect_drawlist_snapshot.py "$RUN_ROOT/checkpoint-147/snapshot/state.bin"
```

Inspector 只驗證外層 envelope，不能取代真正 Load 的語義檢查與還原後讀回。
Program blob 含程序內的 uniform locations，跨程序比較時須先處理 location 對應，
不能把所有 program blob SHA 相同當成必要條件。

執行真正的小型三筆相依 draw 測試：

```bash
python3 script/run_drawlist_checkpoint_integration.py \
  --mesa-prefix "$LLVMPIPE_PREFIX" \
  --renderdoc-source "$RENDERDOC_SOURCE_ROOT" \
  --renderdoc-lib "$CAPTURE_RENDERDOC_LIBRARY" \
  --player "$DRAWLIST_PLAYER" \
  --replay-renderdoc-lib "$SNAPSHOT_RENDERDOC_LIBRARY" \
  --outdir "$RUN_ROOT/tiny-checkpoint-color"
```

此流程包含不中斷執行到 2，以及 save 0 → 新程序 resume 1 → 新程序 resume 2，
各階段 RGBA8 都與獨立 CPU oracle 比對。另用新 output 目錄加 `--depth`，驗證跨 draw
保留的 shared depth attachment；另加 `--hidden-mip` 驗證 draw 0 產生但當下被
`GL_MAX_LEVEL` 隱藏的 mip，能在 restore 後被 draw 1 正常取樣。

Fixture 在原始 capture 的尾端覆寫早期的中間 texture，因此 OpenCapture 的全段預覽
不能替代 checkpoint restore。每筆 draw 的觀察值在原始執行當下保留；resume 結果仍須
符合獨立 CPU oracle。Native fixture 另檢查不中斷執行有 3 筆實際 draw，各段 save／resume
各有 1 筆，且資源保存／載入期間沒有額外 draw 或 dispatch。

若要測 native backend，再加：

```bash
--backend pvrgpu --replay-mesa-prefix "$MESA_PREFIX" --bridge "$NATIVE_BRIDGE"
```

產生原始 capture 的 library 可與 snapshot replay library 不同。省略 `--player` 時只會
產生並驗證 capture，不執行 checkpoint 測試，也不代表 snapshot／resume 已通過。
