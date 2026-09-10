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

預設使用 `third_party/renderdoc-drawlist-snapshot-v2.lock.json` 與
`third_party/renderdoc-drawlist-snapshot.patch`。V2 包含 D16／RGB9_E5 codec，以及 Car Chase
所需的 patch-parameter API aliases／multiview discard attachment 修正。
Builder 只接受 lock 中的原始、V1 或 V2 source bytes；在私有 overlay 正規化後，
重編包含 `gl_debug.cpp`、`gl_common.cpp`、`gl_renderstate.cpp` 在內的八個 translation units。保留的 V1 patch／lock
只供精確辨識與重現，不能將任意 source 改動視為相容。

共用 patch 也以單次、保序分流處理 replay-time validation messages，避免大量訊息
逐筆 erase 的陣列搬移成本。這不關閉 API validation：與原邏輯相同，replay-time
訊息取代 capture 內記錄的訊息，已定 event 與待定 event 的訊息、欄位和順序均保留。

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
不要僅依 capture header 強制提高版本：Car Chase 使用已驗證的 GLES3.1＋
所需 extensions；共用 library 的 capability 檢查不代表 driver 已支援所有 GLES3.2 功能。
此選項只設定 replay context 的版本，不會補上功能，也不能代替 RenderDoc
的 API alias 修正。Car Chase 需要 V2 的 D16 renderbuffer codec；V1 不支援。
小型 fixture 通過不等於真實 capture 的所有格式、狀態及保存邊界都已驗證。

Aztec Ruins Normal 的 capture 位於
`GPU_TestPatterns/7.gl_5_normal/recorder/trace/gl_5_normal_capture_1.rdc`。
選用對應的 Aztec runtime，保存與接續均指定 `--gles-version 3.2`。
Native Mesa 必須能無損執行下述相容 32-bit texture copy；不可只換 RenderDoc
library，就假設所選 Mesa prefix 已具備此能力。Normal 的驗證不能推論 High
或整個 frame 的所有邊界皆受支援。

Version override 不等於 driver 實際具備全部 GLES3.2 功能。共用 replay 在建立
validation controller 前，檢查 texture-buffer、indexed blend／color-mask 的必要
state queries。只有對應 query 明確回報 `GL_INVALID_ENUM` 才視為功能不可用；
既有錯誤、其他錯誤及缺失的必要 entry point 都使初始化失敗，不可清掉後繼續。
只有可由 unindexed state 完整表示的狀態才允許使用既有 scalar 路徑。
非零 texture-buffer binding 或無法表示的獨立 blend／color masks 必須在還原前
拒絕；snapshot Load 也在寫入任何 resource 之前檢查 context。實際 capture 的 API
commands 仍正常執行與稽核，不能因 capability 檢查就略過不支援的指令。

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

Snapshot ABI 與 runtime identity 是兩套不同的版本號：

- Snapshot V1：archive magic `RDGLSN01`，只搭配原有 V1 library／player。
- Snapshot V2：archive magic `RDGLSN02`，加入 D16、typed FBO attachment references，
  以及 renderbuffer binding／pixel-pack direction 狀態；需用 V2 runtime 重新保存。
- Wrapper／新版 player 可辨識兩個 ABI，但 archive、manifest、library 回報及 receipt
  必須一致。V2 library 不載入 V1 archive，也沒有自動升級或只改 manifest 的捷徑。
  `--allow-model-change` 不放寬 ABI 檢查；舊 checkpoint 與其 frozen runtime 應一起保留。

舊 Aztec 專用 RGB9_E5／D24 codec 的 checkpoint 也是 V1，且 D24 編碼與共用 V2
不同。整合後必須從 capture 重新建立 V2 checkpoint，不可改 magic、manifest 或
runtime hash 冒充升級。即使前後皆為 ABI2，換成新增 RGB9_E5 的 library／player
仍屬 runtime identity 改變，不能直接載入舊 binary 所保存的 checkpoint。

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

Snapshot 使用初始內容 codec 還原 supported capture-live objects，並檢查新 controller 的資源
allocation／program identity 能與 checkpoint 相容。無法完整重建的情況必須拒絕，不能
忽略未知物件、遺失 alias 關係，或用零填充補齊。

V2 額外支援 single-sample `GL_DEPTH_COMPONENT16` renderbuffer：

- 保存實際 width × height 的每個 U16 depth code，使用明確 little-endian 編碼，
  不經 PNG、浮點轉換或 shader。
- 載入時以 lossless raw D16 texture staging／同尺寸 depth blit 回原 renderbuffer；
  不換掉 GL object，不拆散兩個 FBO 對同一 renderbuffer 的 alias。
- FBO attachment 同時驗證 ResourceId 與 namespace。所有 archive 資源先完成型別、
  格式、尺寸、samples、長度及引用檢查，才開始還原資源內容。
- 暫時使用的 FBO、texture、PBO／pack／unpack、scissor 與 row-direction 狀態必須復原；
  還原後重新讀取 depth bytes，再做整份 state 驗證。

這不是通用 renderbuffer 支援：其他 depth／stencil／color renderbuffer 格式、MSAA、
layered／multiview／virtual-sample renderbuffer attachment 仍拒絕。
D16 的 width／height 各不超過 32768；包含 120-byte header 的完整 D16 blob
不得超過 256 MiB，超限即拒絕。
`glInvalidateFramebuffer` 不代表物件已不存在；只要仍在 capture-live inventory 內，
就必須保存，不能因當下未使用而略過。

V2 的 D24 texture restore 另外處理 normalized U32 傳輸的非互逆問題：
Mesa 的 bit-expanded U32 readback 若原值直接上傳，經浮點轉換可能少 1 個 D24 code。
Archive 保留原始讀回 bytes；preflight 檢查每個 subresource 的 canonical 表示，
只有 upload scratch 改成對應同一 D24 code 的整數代表值，再由完整 raw readback 確認無損。
此處不允許誤差容忍、捨棄低位，或針對某個 capture／ResourceId 特判。
修正只在 snapshot-active 的 `GL_DEPTH_COMPONENT24` 路徑使用，不更動普通 RenderDoc
OpenCapture 的初始內容處理，也不代表 capture 的原始初始化或 prefix 畫面已通過功能認證。

RGB9_E5 是具有共享 exponent 的 packed 32-bit texture，不能直接當作可渲染的
color attachment 讀取。保存時，以 `glCopyImageSubData` 將同一 physical mip／face／layer
的原始位元複製到相同 target 的 R32UI staging，再用 `RED_INTEGER／UNSIGNED_INT`
讀回。載入使用匹配的 `RGB／UNSIGNED_INT_5_9_9_9_REV` packed upload；不得解成
浮點再重新編碼，因為不同 packed bits 可能代表相同 RGB 值。仍須完整 byte-exact
readback 驗證，不能只比較取樣後的顏色。

此 raw codec 支援 2D、3D、2D array、cube 六面及 cube array；查核 internal format、
尺寸與完整 subresource bytes。Staging 與 archive 受 256 MiB 限制，cube staging
的六面合計也計入上限。臨時 integer texture 使用 nearest 以滿足 copy completeness，
原 texture 的 binding／sampling parameters 不改；FBO、PBO、pack 與 row directions
在成功或錯誤返回時均復原，不使用 helper draw、dispatch 或取樣轉換。

PvrGPU 的 `resource_copy_region` 另支援明確 `VIEW_CLASS_32_BITS` 名單中的
相容格式 raw copy，並保留 target、mip、samples、bounds 檢查。Mesa 也可能將
`glCopyImageSubData` 導向使用相同 R32UI view 的既有 raw blit；需依實際 driver log
辨識路徑，不能把 live copy 成功當成某一個新增 branch 已執行的證據。
這是 Gallium resource transfer 支援，不是 PCO compiler、shader ISS 或 model 的修改。可用
`bash script/run_mesa_resource_unit.sh view-copy` 檢查雙向 RGB9_E5／R32UI copy；
另以 `tests/rgb9e5_texture_transport_live_probe.c` 在實際 Mesa backend 檢查 packed
upload、raw copy、整數 readback，覆蓋 shared exponents、mantissa 及非 canonical bits。

Buffer 的 storage identity 與可變 replay metadata 必須分開：同一個 buffer 可以先綁成
`GL_DRAW_INDIRECT_BUFFER`，後來再綁成 `GL_SHADER_STORAGE_BUFFER`，並同時保留
indexed uniform／vertex 等引用。還原其最後使用 target 時，不得換掉 GL object、
重播 prefix、重新配置 storage，或放寬實際 bytes／尺寸檢查。
真正的 buffer 大小改變仍屬未支援的 allocation 差異，不能與 target 切換混為一談。

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
python3 -m unittest discover -s tests -p test_drawlist_checkpoint_integration.py
python3 -m unittest discover -s tests -p test_drawlist_snapshot_rejection.py
```

Builder tests 另編譯並執行實際的 debug-message 分流與 capability preflight 邏輯。
需要具有 C++ standard-library headers 的 compiler；以 `PVRGPU_SNAPSHOT_TEST_CXX`
明確指定，或沿用 `CXX`。加 `PVRGPU_SNAPSHOT_TEST_SANITIZERS=1` 可執行 ASan／UBSan
版本。找不到 headers 或編譯失敗不是測試成功，也不應以跳過檢查代替。

可以用只讀 inspector 列出 archive 的資源 ID、signature digest 與內容 digest：

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

另外以新 output 目錄加 `--depth-renderbuffer`，執行獨立的 7×5 D16 fixture：
兩個 FBO 共用一塊 depth，draw 1 改寫 depth，draw 2 的通過／拒絕結果必須依賴
這次更新。每個邊界與 CPU oracle 比對，最後整份 archive 與不中斷執行逐 byte 比對。
此選項不能與 `--depth` 或 `--hidden-mip` 合用。

以 `--depth-low-codes` 另開一個 7×5 D24 texture fixture，保存 depth codes
1／159／256／65535，並讓後兩筆 draw 依賴共享附件的新 depth 值。Producer 額外直接讀取
每個邊界的 U32 depth，與獨立整數 oracle 比對；checkpoint 仍須通過完整 state 比對。
此模式與其他 depth／hidden-mip 選項互斥，用來偵測一般深度漸層可能漏掉的 1-LSB 流失。

`--buffer-target-alias` 則使用同一個 32-byte UBO，保留 shader 所用的 indexed uniform
binding，另外切換其 generic indirect／SSBO target，並在 capture 尾端改成 COPY_READ。
大小與 GL object 不變；原始 capture 查詢會核對共享 binding，接續後仍須符合 CPU 色彩
oracle 與完整 archive。此模式不可與其他 fixture 模式合用。

`--rgb9e5` 使用兩塊僅供取樣的 packed RGB9_E5 texture，各含兩個 mip，並讓後兩筆
draw 依賴先前保存的 packed texture 與 RGBA8 中間結果。使用可由整數 oracle 計算的
顏色，以及 RGB 浮點值相同但 packed bits 不同的表示；除了色彩 oracle，還要核對
archive 中的 packed 原始內容及最終整份 archive。此模式與其他 fixture 模式互斥。

Fixture 在原始 capture 的尾端覆寫早期的中間 texture，因此 OpenCapture 的全段預覽
不能替代 checkpoint restore。每筆 draw 的觀察值在原始執行當下保留；resume 結果仍須
符合獨立 CPU oracle。Native fixture 另檢查不中斷執行有 3 筆實際 draw，各段 save／resume
各有 1 筆，且資源保存／載入期間沒有額外 draw 或 dispatch。
除了 driver accepted-draw events，也累計 model 的實際 `counters.drawlists`；
不能只靠 replay 的邏輯範圍宣稱省略了 prefix。

若要測 native backend，再加：

```bash
--backend pvrgpu --replay-mesa-prefix "$MESA_PREFIX" --bridge "$NATIVE_BRIDGE"
```

產生原始 capture 的 library 可與 snapshot replay library 不同。省略 `--player` 時只會
產生並驗證 capture，不執行 checkpoint 測試，也不代表 snapshot／resume 已通過。

## V2 codec 錯誤注入

對已通過上述 D16 正向測試的 `save0/snapshot` 執行真正的 Load 拒絕測試：

```bash
python3 tests/test_drawlist_snapshot_rejection.py run \
  --snapshot-dir "$RUN_ROOT/tiny-checkpoint-d16/save0/snapshot" \
  --capture "$RUN_ROOT/tiny-checkpoint-d16/capture/three-draw-dependency_capture.rdc" \
  --player "$DRAWLIST_PLAYER" \
  --renderdoc-lib "$SNAPSHOT_RENDERDOC_LIBRARY" \
  --mesa-prefix "$LLVMPIPE_PREFIX" --backend llvmpipe \
  --through-draw 1 --outdir "$RUN_ROOT/tiny-checkpoint-d16-rejection" \
  --codec-source "$NEW_SNAPSHOT_BUILD_DIR/source/renderdoc/driver/gl/gl_replay_snapshot.inl" \
  --safety-source "$NEW_SNAPSHOT_BUILD_DIR/source/renderdoc/driver/gl/gl_snapshot_codec_safety.inl"
```

Native 測試必須使用 native 正向測試的 checkpoint，並改用其 Mesa prefix／backend／bridge。
測試會複製 archive、改動 D16／FBO 內層欄位，再重算外層 checksum 及測試用 manifest／receipt，
確認不是只有 wrapper 的 hash 檢查有效。原始 checkpoint 不改動。

每個 case 必須在新的 process 由真正 codec 以預期原因拒絕，不執行 suffix、不產生新 checkpoint
或成功 receipt；native interval 不得產生 model 工作。測試另核對 source 中「全資源 preflight
先於 restore loop」的順序，但這是 source-order 證據，不能把缺少 receipt 當成動態證明
所有 GL resource 都完全未寫入。測試中重新簽封的錯誤 archive 絕非可用 checkpoint。

另外使用 `--depth-low-codes` 正向測試的 seed 加 `--depth24-mode`，檢查第一／最後
D24 DWORD 的非法低位表示；使用 `--buffer-target-alias` seed 加 `--buffer-target-mode`，
檢查非法 target、改動 physical size、改動 cached size。三種負例模式互斥，各自使用
對應 capture、backend 與已驗證的 checkpoint；不得混用不同 fixture 的 seed。
