# PDS (Programmable Data Sequencer) — 模組分析與 OpenGL ES 3.x 補齊規劃

## 1. 架構概述

在 PowerVR TBDR 架構中，**PDS (Programmable Data Sequencer)** 是一個非常關鍵的硬體微處理器。它不參與複雜的浮點數運算，而是專職作為 **USC (Unified Shading Cluster) 的前導調度官**。

PDS 的主要功能包含：
1. **資料準備 (Data Ingest & Packing)**：自記憶體/快取加載頂點屬性、常數 (Uniforms) 或平面方程式係數，並打包填寫到 USC 的暫存器。
2. **任務分派 (Task Generation & Scheduling)**：透過執行內建的 PDS 程式，計算執行緒遮罩與執行寬度，並向 USC 發布任務描述符（Task Descriptors）啟動 Shader 執行。
3. **資料輸出指示 (DOUT / DOUTI)**：發出 DOUTI 數據傳輸信號，通知 USC 哪些暫存器已被初始化完畢。

目前 PvrGPU 包含兩個 PDS 引擎：
- **`PdsEngine` (Fragment PDS)**：已實作完整的片段平面方程式係數 (Varying Coefficients) 載入與任務分派。
- **`VertexPdsEngine` (Vertex PDS)**：目前為 Pass-Through 結構預留點。

---

## 2. PdsEngine (Fragment) — 核心邏輯與偽碼

在片段階段，`PdsEngine` 將光柵化生成的像素方塊 (2x2 Quads) 與三角形的插值平面參數（Plane Coefficients）相結合，打包為 USC 能夠理解的 Task。

### 偽碼 (Pseudo-code)

```python
# 模擬 2x2 Fragment Quad 任務打包與係數加載
Process PdsEngine::Run():
  while true:
    txn = input.read()  # 接收來自 FragmentFrontend (Rasterizer) 的通知
    state = LoadPipelineState(pool, txn.state)
    assert state.stage == PipelineStage::kFragmentsReady

    # 1. 載入光柵化生成的數據
    invocations = LoadArray(pool, state.fragment_invocations)
    quads = LoadArray(pool, state.fragment_quads)
    parameters = LoadArray(pool, state.parameter_triangles) # 三角形 Edge & Plane 方程式

    # 2. 判斷 Varying 係數載入屬性
    varying_case = UsesShaderVaryings(state.functional_case)
    coefficient_dwords = VaryingCoefficientDwordCount(state.functional_case)
    
    if varying_case:
      parameter_coefficients = LoadArray(pool, state.parameter_coefficients)

    tasks = []
    usc_coefficient_banks = [] # 準備寫入 USC 的暫存器堆

    # 3. 遍歷所有的 Quad，為每個 Quad 分配執行緒任務
    for quad_index, quad in enumerate(quads):
      # 驗證 Coverage Mask 與頂點關聯性
      validate_quad_metadata(quad, parameters)

      # 建立 USC Task 描述符
      task = UscFragmentTask()
      task.fragment_quad_index = quad_index
      task.first_coefficient_dword = len(usc_coefficient_banks)

      if varying_case:
        task.coefficient_dword_count = coefficient_dwords
        
        # 依據 Quad 對應的三角形載入 Varying 面方程式參數 (A, B, C 加上 Pad 共 4 dwords/varying)
        triangle = parameters[quad.parameter_index]
        start_set = triangle.first_coefficient_set
        
        for set_idx in range(triangle.coefficient_set_count):
          coef = parameter_coefficients[start_set + set_idx]
          usc_coefficient_banks.append(coef.a)   # 平面方程式係數 A
          usc_coefficient_banks.append(coef.b)   # 平面方程式係數 B
          usc_coefficient_banks.append(coef.c)   # 平面方程式係數 C
          usc_coefficient_banks.append(0)        # Align 填零 (Pad)

      tasks.append(task)

    # 4. 回寫到 MemoryPool 中，供 USC Cluster 的 Fragment Shader 直接讀取
    state.usc_fragment_tasks = StoreNewArray(pool, tasks)
    state.usc_coefficient_banks = StoreNewArray(pool, usc_coefficient_banks)
    
    # 5. 更新 PDS 指令計數器
    if varying_case:
      state.counters.pds_coefficient_tasks = len(tasks)
      state.counters.pds_douti_issues = len(tasks) * 2  # 模擬發送兩次 DOUTI 訊號
      state.counters.usc_coefficient_load_bytes = len(usc_coefficient_banks) * 4

    state.stage = PipelineStage::kPdsReady
    StorePipelineState(pool, txn.state, state)
    output.write(txn)  # 傳遞至 FragmentSlot/USC
```

---

## 3. VertexPdsEngine (Vertex) — 核心邏輯與偽碼

在頂點階段，`VertexPdsEngine` 作為 pass-through 結構預留點。

### 偽碼 (Pseudo-code)

```python
Process VertexPdsEngine::Run():
  while true:
    txn = input.read()  # 接收來自 VertexFetch 的通知
    state = LoadPipelineState(pool, txn.state)
    assert state.stage == PipelineStage::kVertexFetched

    # TODO: 模擬實體頂點索引/屬性 PDS 定序與加載
    # 1. 讀取頂點 attribute 描述與 index buffer
    # 2. 為每個頂點執行緒封裝 USC Task Descriptor (分配輸入暫存器)
    # 3. 發送 DOUTI 指令以初始化 VS 暫存器
    
    state.stage = PipelineStage::kVertexPdsReady
    StorePipelineState(pool, txn.state, state)
    output.write(txn)  # 傳遞至 PcoDecoder
```

---

## 4. 與 OpenGL ES 3.x 功能缺失對照

### 4.1 Fragment PDS

| OpenGL ES 3.x 功能 | 目前狀態 | 缺失說明 | 補齊路徑 |
|-------------------|---------|---------|---------|
| 平面方程式插值多樣化 (Interpolation Qualifier) | ❌ 硬編碼 | 目前僅支援對 $1/W$ 做常規透視校正插值，忽略 `flat` 或 `noperspective` | PDS 需要解析 PCO 欄位，控制 `flat` 模式下直接將 Provoking Vertex 數值拷貝，而非帶入 ABC 平面公式 |
| 質心插值 (Centroid) | ❌ 不支援 | 無法針對多重採樣邊緣的 `centroid varying` 重新校正插值座標 | 需由 PDS 依據 MSAA 覆蓋遮罩動態微調插值中心點 |

### 4.2 Vertex PDS（目前為 Stub）

| OpenGL ES 3.x 功能 | 目前狀態 | 缺失說明 | 補齊路徑 |
|-------------------|---------|---------|---------|
| 頂點實例化繪製 (Instancing) | ❌ 不支援 | 無法在頂點暫存器中為頂點著色器初始化並填入 `gl_InstanceID` | PDS 需維護當前實例計數，並動態將 `InstanceID` 寫入 USC 預留的輸入暫存器中 |
| 常數緩衝區自動分發 (UBO) | ❌ 不支援 | 目前 Uniforms 是由 CPU 模擬器直接寫入共享暫存器，沒有經過 PDS 定序器的加載通道 | 實作 PDS Uniform-Fetch 邏輯，藉由模擬讀取 SLC 將 UBO 位址的資料透過 DOUT 加載至暫存器 |

### 4.3 Compute PDS（待建立）

| OpenGL ES 3.1 功能 | 目前狀態 | 缺失說明 | 補齊路徑 |
|-------------------|---------|---------|---------|
| Compute Task 分派 | ❌ 缺失 | 缺乏對 `glDispatchCompute` 執行緒工作群組 (Workgroups) 的三維劃分與調度 | 新建 `ComputePdsEngine`，將工作群組劃分為複數執行緒 Task，發送至 USC 進行並行計算 |

---

## 5. 漸進式實作路線圖

### Phase 1：頂點實例化 (GLES 3.0 Instancing) 與 Flat Shading
- [ ] **頂點實例 PDS 支援**：在 `VertexPdsEngine` 中解析 `DrawCommand` 資訊，將 `gl_InstanceID` 依執行個體步進計數寫入 Task 暫存器映像中。
- [ ] **Flat Shading 欄位傳遞**：PdsEngine 解析 Varying 繫結。當偵測到 `flat` 限定符時，停止平面 ABC 插值計算，直接傳送基準頂點的 constant color。

### Phase 2：MSAA 質心插值與 UBO 模擬
- [ ] **UBO 定序加載**：擴充 PDS 以支援對 Uniform Buffer 的虛擬讀取，將記憶體常數打包為 USC 常數暫存器。
- [ ] **Centroid 插值修正**：片段 PDS 依據覆蓋率微調插值係數。

### Phase 3：Compute PDS 實現 (GLES 3.1)
- [ ] **Compute PDS 模組實作**：對接 `ComputeDataMaster`，處理 3D Thread Dispatch 並指派任務至 USC Cluster。
