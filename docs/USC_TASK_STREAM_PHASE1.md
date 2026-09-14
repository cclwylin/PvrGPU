# USC Task Stream — Phase 1 介面設計（Unified Shader 前置）

狀態：設計草案，待確認後實作。
範圍：VS / GS / PS 統一執行與輸出介面。TCS / TES / CS 在 Phase 2 接入。

## 1. 目標

1. **統一介面**：VS、GS、PS 走同一套 shader task 介面，包含
   - task 發派（每組 `usc_issue_lanes` = 4 lanes）
   - 執行、暫停與恢復
   - 輸出串流
   - 時序與統計

   Phase 2 可在同一介面下合併各 stage 的 ISS，Phase 3 再合成單一 USC 排程器。
2. **GS 輸出改為有容量的串流**。輸出達到 parameter buffer 容量時觸發 partial render，
   解決 `stress.tessellation_geometry_interaction.render_multiple_limits.output_max_required`：
   3,355 萬個輸出 vertex，目前一次載入約需 17 GiB host 記憶體。
3. **VS / PS 功能完全不變**。輸出影像與現有 counter 必須逐位元相同，由 regression suite 與 ctest 保證。

非目標（Phase 1 不做）：
- 合併 ISS 指令語意
- 改變 VS / PS 的 texture 批次形狀
- TCS / TES / CS 改走新介面

## 2. 現況

| | VS | PS | GS |
|---|---|---|---|
| 執行模組 | `UscSlot → UscCluster` | `FragmentFrontend → PdsEngine → UscSlot → UscCluster` | 獨立 `GeometryShader` |
| ISS | `pco_iss` Execute/Resume（continuation） | `pco_iss` Execute（context + continuation） | `geometry_iss` Step（blocking callback） |
| Lane 分組 | 4-lane group 只用於時序計算；功能上整個 draw 一起 lock-step | 2×2 quad、每批 256 quad 常駐 | 單一 lane，逐個 primitive × invocation |
| SMP | 所有 lane 一批進 texture FIFO | 依 (descriptor set, PC) 分批 | 每次 SMP 各跑一趟 FIFO |
| 輸出 | 原地寫回 `vertex_lanes` | 新配置 `fragment_outputs` | 全部累積完才一次發佈 lanes / refs / primitives |
| 時序 | `2+⌈g/2⌉`（slot）、`4+⌈g/4⌉`（cluster） | 同左，計入 `renderer_cycles` | `group_count + instructions`，不計入 `tiler_cycles` |

`PcoTextureRequest → TextureSampleRequest` 的轉換重複實作了 6 次，
DrawList 統計也各寫各的。

## 3. 統一介面（`src/systemc/shader/usc_task_stream.{h,cpp}`）

這是一般 C++ 函式庫，**不是 sc_module**，以符合 `check_systemc_module_layout`。

```cpp
enum class UscStage : uint8_t { kVertex, kGeometry, kFragment /* Phase 2: kTessControl, kTessEval, kCompute */ };

// 一條 lane 的身分；不同 stage 只填自己用得到的欄位。
struct UscLaneIdentity {
  uint32_t task_id;          // VS: lane; GS: primitive×invocation 序號; PS: quad task
  uint32_t global_lane;      // 送 texture 時的 shader_lane_index
  uint32_t quad_id; uint8_t quad_lane; uint8_t sample_id; uint8_t helper;
};

// 發派計畫：lanes → 每組 kReferenceUarch.usc_issue_lanes 條。
struct UscIssuePlan {
  static UscIssuePlan ForLanes(UscStage, uint64_t lanes);   // groups = ⌈lanes/4⌉
  uint64_t lanes, groups;
};

// 單一 lane 的執行結果（包裝各 ISS 原本的回傳型別）。
enum class UscLaneStatus : uint8_t { kCompleted, kSuspendedTexture, kSuspendedDerivative };

// 各 stage 的 adapter：只轉接既有 ISS 入口，不改指令語意。
class UscStageExecutor {
 public:
  virtual UscLaneStatus Issue(uint32_t lane) = 0;                          // 第一次執行
  virtual UscLaneStatus ResumeTexture(uint32_t lane, const uint32_t rgba[4]) = 0;
  virtual const PcoTextureRequest &PendingTexture(uint32_t lane) const = 0;
  virtual void Emit(uint32_t lane, UscOutputSink &sink) = 0;               // 完成後輸出
};
// VertexExecutor   → ExecuteVertexPco / ResumeVertexPco
// FragmentExecutor → ExecuteFragmentPco(prepared, ctx + continuation)
// GeometryExecutor → StepGeometryTask 迴圈（Phase 1 的 SMP 仍用同步 callback）

// 共用的 texture request 建構（取代重複的 6 份）。
TextureSampleRequest MakeUscTextureRequest(UscStage, const PcoTextureRequest &,
                                           const UscLaneIdentity &, uint32_t request_id);

// 共用的指令統計：DrawList 與 vs_* / gs_* / fs_* counter。
class UscStatsRecorder { /* 由 RecordInstructionExecutions 抽出 */ };

// 時序模型：每個 stage 一張速率表，單位是 cycle。
struct UscStreamRates {           // 每單位工作花幾個 T
  uint32_t accept_per_input;      // 輸入 primitive / vertex
  uint32_t per_invocation;
  uint32_t per_emitted_vertex;
  uint32_t per_exported_primitive;
};
inline constexpr uint32_t kUscStreamPeriodCycles = 1;   // T = 1 cycle
inline constexpr UscStreamRates kGeometryStreamRates = {1, 2, 1, 2};   // 使用者指定

// 有容量上限的輸出串流；滿了就呼叫 flush。
template <class Record> class UscOutputStream {
 public:
  UscOutputStream(uint64_t capacity, std::function<void(std::vector<Record>&&)> flush);
  void Push(Record);            // 到達 capacity 時呼叫 flush
  void Close();                 // 送出最後一批（final）
};
```

### 各 stage 在 Phase 1 的接法

- **VS**：`UscCluster` vertex branch 改用 `VertexExecutor`、`MakeUscTextureRequest`、`UscStatsRecorder`、`UscIssuePlan`。
  - SMP round 仍是「所有 lane 一批」，輸出仍原地寫回 `vertex_lanes`。
  - 輸出 sink 是 in-place sink：VS 的輸出容量等於整個 draw，不分批。
- **PS**：`UscCluster` fragment branch 改用 `FragmentExecutor` 和共用 request/stats。
  - 256 quad 常駐、derivative rendezvous、helper lane 規則都不變。
  - 輸出 sink 寫進 `fragment_outputs`。
- **GS**：`GeometryShader::Execute` 重寫在介面上：
  - invocation 以 4 條 lane 一組發派，只影響時序與統計，執行仍是單 lane Step。
  - emit 的 vertex 與 assemble 出的 primitive 進 `UscOutputStream`，容量為 `kGeometryStreamCapacityPrimitives`。

## 4. GS 串流與 Partial Render

### 4.1 容量

- `kGeometryStreamCapacityPrimitives = 1 << 20`（約 104 萬個 primitive）。
- 以每個 primitive 在下游約 1 KB 估算（VertexLane 516 B、RasterTriangle 120 B、ParameterTriangle 232 B，再加 coefficients 等），每批約 1 GiB。
- 放在 `reference_uarch.h`，意義是 parameter buffer 容量。

輸出超過容量才會分批。其他所有 draw 只有一批，行為和 counter 都不變。

### 4.2 協定（新增 FIFO：`PbeWriteBack → GeometryShader` 的 `geometry_partial_render_done`）

```
GS: Execute 開始
    保存「原始 LOAD 證據」(color/depth load handle、enable、bytes、attachment_clears)
    counters 累加器 A = state.counters；state.counters 清零（frame 欄位保留）
    loop 輸入 primitive → invocation（每組 4 lanes）:
        Push primitives 進 stream；stream 滿 → flush(batch, final=false)
    stream.Close() → flush(last, final=true)

flush(batch, final):
    發佈 batch 的 lanes / refs / geometry_primitives
    state.partial_render = { batch_index, final }
    final=true 時：A 存入 state.partial_render_counters handle
    output.write(txn)                     ── 下游 ClipCull … PbeWriteBack 照常處理
    final=false:
        wait geometry_partial_render_done.read(txn)
        A += state.counters（依欄位合併規則，見 4.4）；state.counters 清零
        ResetForNextPartialRender(state)   ── 見 4.3

PbeWriteBack 結束時:
    partial_render.final == 0 → 寫到 geometry_partial_render_done（不送 reporter）
    final == 1 且有 partial_render_counters →
        counters = merge(A, counters)；還原「原始 LOAD 證據」；送 reporter
```

reporter 只會看到一個 txn，LOAD 證據也是原始值，所以 `json_reporter` 不需要修改。

### 4.3 `ResetForNextPartialRender`（讓下一批像一個新的下游 draw）

1. **Color**：`dram_framebuffer`（加上 MRT extras）轉成 `color_attachment_load`，設 `_enable=1`、`_bytes`，並釋放 dram payload。
   依 submitter 做法用 `memory_->Readback(kFramebufferReadback)` 讀回，讓 DRAM 讀取計入 counter。
2. **Depth / stencil**（`depth_format != 0` 時）：`depth_attachment` 轉成 `depth_attachment_load`，並清掉 `depth_attachment_ready` 和 `isp_depth` / `isp_stencil`。
3. **Clear**：拿掉 `attachment_clears`。clear 只在第一批套用。
4. **下游中間產物全部釋放並清成空 handle**：
   - `raster_triangles`、`raster_vertex_outputs`
   - `tile_records`、`tile_primitive_refs`、`scheduled_tiles`
   - `fragment_candidates`、`fragment_invocations`、`fragment_shader_lanes`、`fragment_quads`
   - `usc_fragment_tasks`、`usc_coefficient_banks`、`fragment_outputs`、`fragment_instructions`
   - parameter pool handles
   - stage 回到 `kVertexShaded`
5. **Stream output**：有 TF target 時若需要分批，Phase 1 直接 fail-closed，錯誤訊息要明確。
   目前的 dEQP 沒有這種組合。

### 4.4 Counter / DrawList 合併規則

- 預設逐欄位加總，例如 `c_primitives`、`setup_triangles`、`covered_pixels`、`isp_cycles`、`pbe_*`、`occlusion_samples_passed`、`dram_*`、`texel_fetches`、`ps_invocations`。
- 例外：
  - `frame`、`functional_frame`、`drawlists` 保留原值。
  - `pool_bytes_in_flight`、`pool_high_water_bytes` 取最大值。
- DrawList：
  - 各 shader stage 的 executed counts 加總；
  - `program_*` 取原值；
  - `executions_recorded` 固定為 1。
- 合併後必須通過 reporter 既有的 `ValidateDrawListStats` 與 texture 對帳：`texture_requests == Σ *_tex_instructions`。

### 4.5 正確性說明

- **等價性**：分批後的 LOAD 串接，等同於照原本 primitive 順序依序畫完。
  blend、depth/stencil test、HSR、occlusion、MSAA、layered framebuffer 都是 attachment 狀態的延續。
  GS 的 primitive ID 與 invocation ID 由 GS 連續產生，不受分批影響。
- **FS 副作用**：image / SSBO 寫入存在 DRAM，跨批次持續有效。

## 5. GS 時序（使用者指定速率，T = 1 cycle）

```
gs_stream_cycles = Σ_input_primitives (1T)
                 + Σ_invocations      (2T)
                 + Σ_emitted_vertices (1T)
                 + Σ_exported_prims   (2T)
```

- **計入**：`usc_cluster_cycles` 與 `tiler_cycles`（geometry phase）。
- **取代**：原本的 `group_count + instructions`。
- **`usc_groups`**：改為 `⌈invocations / 4⌉`，與 VS 相同定義。原本是 `+= instructions`。
- **`WaitForCycles`**：每批 flush 前等待該批的 stream cycles。這樣下游收到每批的時間點，符合 export 速率。
- **影響範圍**：只影響有 GS 的 draw 的時序 counter。
  - glbench / glmark2 / Manhattan 沒有 GS，counter 不變。
  - dEQP 只看影像。
  - `geometry-shader-unit` 與 `pco-geometry-api-bridge-unit` 裡釘住 GS counter 公式的測試需要同步更新。

## 6. 不變條件

- **VS / PS 輸出與 counter 逐位元不變**：
  - VS 每個 SMP round 仍是一批所有 lane，`request_id` = lane。
  - PS 的 texture batch 仍依 (set, PC)，256 quad 常駐，implicit LOD 批次維持 quad 對齊。
- **host byte 預算**：`TEXTURE_RESIDENT_HOST_BYTES` 不變，request / continuation 結構大小不變。
- **helper lane 規則**：helper lane 不輸出、無副作用；derivative 需要完整 quad。
- **timing / pool**：每個 USC txn 只 `WaitForCycles` 一次（GS 為每批一次）；MemoryPool 保持平衡。
- **GS backpressure**：`nb_read` / `nb_write` 行為維持（`geometry_shader_test` 770-786）。

## 7. 檔案改動清單

| 檔案 | 改動 |
|---|---|
| `src/systemc/shader/usc_task_stream.{h,cpp}`（新） | 介面、adapters、request builder、stats recorder、rates、output stream |
| `src/systemc/shader/usc_cluster.cpp` | VS / PS branch 改用新介面（不改行為） |
| `src/systemc/shader/geometry_shader.{h,cpp}` | 串流、4-lane 發派、partial render flush、新 FIFO port |
| `src/systemc/fragment/pbe_write_back.{h,cpp}` | partial render 回送 GS、final 合併 counters、還原 LOAD 證據 |
| `src/systemc/common/pipeline_state.h`、`functional_types.{h,cpp}` | `partial_render` 欄位；`MergePartialRenderCounters` / DrawList 合併；`ResetForNextPartialRender` |
| `src/systemc/common/reference_uarch.h` | `geometry_stream_capacity_primitives`、stream period |
| `model_stub/pvrgpu_model_stub.cpp` | 接上 `geometry_partial_render_done` FIFO |
| `CMakeLists.txt` | 約 15 個 target 加入 `usc_task_stream.cpp` |
| tests | 新增 `usc-task-stream-unit`、`geometry-partial-render-*`；更新 GS counter 公式相關測試 |

## 8. 驗證計畫

1. **單元測試**：
   - `usc-task-stream-unit`：issue plan、request builder、output stream 容量與 flush 次數。
   - `geometry-partial-render-*`：用測試 hook 把容量壓到很小，強制分成 N 批。
     影像與 depth 必須和單批完全一致，counter 必須等於單批加總。
     另含 TF+分批的 fail-closed 案例、MRT 案例、stencil 案例。
2. **ctest 全部**：只允許既有的 `systemc-module-layout` 失敗。
3. **Regression suite**（glbench、glmark2 80×60 / 800×600、Manhattan capture_1）：counter 與影像和 baseline 相同。
4. **dEQP**：
   - `geometry_shading.*`、`tessellation_geometry_interaction.*`、`stress.*` 全跑。
   - 目標 case 在 1800 s timeout 內 Pass，並記錄峰值記憶體。
5. **L4**：從中斷處續跑。

## 9. 待確認

1. 容量 `1<<20` primitives / 批：可接受，或要改成其他值？例如依 PB 記憶體 MiB 換算。
2. `T = 1 cycle`：可接受？
3. GS 的 `usc_groups` 改為 `⌈invocations/4⌉`、`usc_cluster_cycles` 改用 stream rates（見第 5 節）：可接受？
4. TF + 需要分批：Phase 1 先 fail-closed，可接受？
