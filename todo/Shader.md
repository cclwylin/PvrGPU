# PvrGPU 著色器管線設計與 OpenGL ES 3.x 補齊規劃

本文件詳細分析 PvrGPU 內部的著色器執行管線（USC / PCO ISS），與 OpenGL ES 3.x 著色器標準規格進行對比，找出指令集與功能性缺失，並建立對應的開發任務矩陣。

---

## 1. PvrGPU 著色器執行架構與流程

PvrGPU 著色器系統模擬了 PowerVR 的 Unified Shading Cluster（USC）著色暫存器堆與指令執行單元。

### 1.1 著色器執行資料流
```
[ Parameter Buffer / VBO ]
           |
           v (Stage: kVertexIssued / kFragmentIssued)
+------------------------------------+
|       UscCluster & Pco ISS         |
+------------------------------------+
|  1. Decode PCO instructions        |
|  2. Load Inputs/Shared Registers   |
|  3. Interpreter Loop (pc index)    |
|  4. Interpolate Varyings (FITRP)   |
|  5. Execute Math/Logic operations  |
|  6. Write outputs (VTXOUT/PIXOUT)  |
+------------------------------------+
           |
           v (Stage: kVertexShaded / kTextureComplete)
[ Rasterizer Setup / PBE ]
```

### 1.2 核心步驟與虛擬碼 (Pseudo-code)

以下為目前 USC 著色器群組模擬器及 PCO 指令集譯碼執行器（ISS）的核心邏輯：

```python
# USC 著色器模擬與指令集解碼執行虛擬碼

# =======================================================
# 1. USC Cluster 頂點與片段著色器排程
# =======================================================
def run_usc_cluster(state: PipelineState, pool: MemoryPool, stage: ShaderStage):
    if stage == ShaderStage::kVertex:
        assert state.stage == PipelineStage.kVertexIssued
        lanes = load_array(pool, state.vertex_lanes)
        instructions = load_array(pool, state.vertex_instructions)
        
        # 逐頂點執行著色器
        for lane in lanes:
            execution = execute_vertex_pco(state.vertex_program_summary, instructions, lane.vertex_input)
            lane.vertex_output = execution.outputs
            lane.emitted = execution.emitted
            lane.ended = execution.ended_task
            
        store_array(pool, state.vertex_lanes, lanes)
        state.stage = PipelineStage.kVertexShaded
        
    else: # Fragment stage
        assert state.stage == PipelineStage.kFragmentIssued
        invocations = load_array(pool, state.fragment_invocations)
        instructions = load_array(pool, state.fragment_instructions)
        quads = load_array(pool, state.fragment_quads)
        tasks = load_array(pool, state.usc_fragment_tasks)
        coefficient_bank = load_array(pool, state.usc_coefficient_banks)
        shared_registers = load_array(pool, state.fragment_shared_registers)
        
        # 以 2x2 Quad 為單位進行調度與執行
        for task in tasks:
            quad = quads[task.fragment_quad_index]
            context = setup_fragment_context(quad, task, coefficient_bank, shared_registers)
            
            for lane in range(4):
                if not quad.active_mask[lane]:
                    continue
                
                # 執行 Fragment PCO 程式
                execution = execute_fragment_pco(state.fragment_program_summary, instructions, context)
                
                # 遇紋理採樣 SMP 指令時，執行緒會掛起 (Suspend) 並保留 Continuation 狀態
                if execution.suspended:
                    save_fragment_continuation(quad.invocation_indices[lane], execution.continuation)
                    add_texture_sample_request(execution.texture_request)
                else:
                    save_pixel_output(quad.invocation_indices[lane], execution.pixel_outputs)

# =======================================================
# 2. PCO ISS 指令解譯執行器 (pco_iss.cpp)
# =======================================================
def execute_fragment_pco(summary: PcoProgramSummary, instructions: list[PcoInstruction], context: PcoFragmentExecutionContext):
    # 初始化暫存器檔案
    temporaries = [0] * kPcoTemporaryCount
    temporary_written_mask = 0
    pixel_outputs = [0] * kPcoPixelOutputCount
    
    pc = 0
    while pc < len(instructions):
        inst = instructions[pc]
        
        # 情況 A: 分支跳轉指令
        if inst.opcode == PcoOpcode.kBranch:
            pc = inst.branch_target_index
            continue
        elif inst.opcode == PcoOpcode.kBranchConditional:
            cond = read_source(inst.source, temporaries)
            if cond != 0:
                pc = inst.branch_target_index
            else:
                pc += 1
            continue
            
        # 情況 B: 紋理採樣掛起
        if inst.opcode == PcoOpcode.kTextureSample:
            # 填充紋理採樣請求包，並回傳掛起狀態
            texture_request = setup_texture_request(inst, temporaries, context)
            continuation = save_continuation_state(pc + 1, temporaries, temporary_written_mask)
            return PcoFragmentExecution(suspended=True, texture_request=texture_request, continuation=continuation)
            
        # 情況 C: 片段拋棄 (Discard)
        if inst.opcode == PcoOpcode.kDiscard:
            return PcoFragmentExecution(discarded=True)
            
        # 情況 D: 常規 ALU 運算指令 (FADD, FMUL, FMAD, FMIN, FMAX, LOG2, SIN, COS 等)
        if is_alu_operation(inst.opcode):
            src0 = read_source(inst.source, temporaries, context.shared_registers)
            result = 0
            
            if inst.opcode == PcoOpcode.kMoveBypass:
                result = src0
            elif inst.opcode == PcoOpcode.kFloatAdd:
                src1 = read_source(inst.source1, temporaries)
                result = float_add(src0, src1)
            elif inst.opcode == PcoOpcode.kFloatMultiply:
                src1 = read_source(inst.source1, temporaries)
                result = float_multiply(src0, src1)
            elif inst.opcode == PcoOpcode.kFloatMad:
                src1 = read_source(inst.source1, temporaries)
                src2 = read_source(inst.source2, temporaries)
                result = float_mad(src0, src1, src2)
            elif inst.opcode == PcoOpcode.kFloatSine:
                result = float_sine(src0)
            elif inst.opcode == PcoOpcode.kFloatCosine:
                result = float_cosine(src0)
            elif inst.opcode == PcoOpcode.kPackHalf2x16:
                src1 = read_source_component(inst.source, component=1, temporaries)
                result = pack_half_2x16(src0, src1)
            elif inst.opcode == PcoOpcode.kUnpackHalf2x16:
                result = unpack_half_2x16_low(src0)
                temporaries[inst.output_index + 1] = unpack_half_2x16_high(src0)
                temporary_written_mask |= (1 << (inst.output_index + 1))
            # 其餘為 Stub 的指令（如 dFdx, dFdy 等）
            elif is_stub_operation(inst.opcode):
                result = src0 # Bypass
                
            temporaries[inst.output_index] = result
            temporary_written_mask |= (1 << inst.output_index)
            pc += 1
            continue
            
        # 情況 E: 輸出至 Framebuffer
        if inst.opcode == PcoOpcode.kMoveBypass and inst.target == PcoWriteTarget.kPixelOutput:
            pixel_outputs[inst.output_index] = read_source(inst.source, temporaries)
            pc += 1
            
    return PcoFragmentExecution(pixel_outputs=pixel_outputs, suspended=False)
```

---

## 2. 與 OpenGL ES 3.x 著色器功能對比分析

目前 PvrGPU 的著色器解碼與執行器僅能支持基本測試案例（Fill.Solid / Varyings / Basic Texture），相較於完整的 OpenGL ES 3.x，在著色語言 (GLSL ES) 特性與管線支援上存在以下缺失：

| 功能分類 | OpenGL ES 3.x 標準規格 | PvrGPU 目前狀態 | 關鍵缺失與硬體落差 |
| :--- | :--- | :--- | :--- |
| **著色器階段 (Stages)** | 頂點 (VS)、片段 (FS)、計算 (CS，3.1)、幾何 (GS，3.2)、細分控制/評估 (3.2) | 僅支援 **VS** 與 **FS** | 缺少 Compute Shader 執行架構，無幾何/細分著色器的控制與暫存器拓撲支援。 |
| **偏導數指令 (Derivatives)**| 支援 `dFdx`, `dFdy`, `fwidth` | 指令存在，但為 **Stub (直通)** | `kDerivativeX` 與 `kDerivativeY` 單純回傳輸入值，未在 2x2 像素方塊（Quad）的相鄰 lane 間進行差值計算，影響自訂 Mipmap 與過濾精度。 |
| **SSBO 與 Image 載入/寫入** | 支援 `layout(binding=N) buffer` 讀寫與 `imageLoad` / `imageStore` (3.1) | 指令存在，但為 **Stub (直通)** | `kBufferLoad` 與 `kBufferStore` 未執行真正的記憶體請求與 SLC/DRAM 交互，無法支持計算著色器的隨機讀寫。 |
| **記憶體原子操作 (Atomics)** | 支援 `atomicAdd` 等快取一致的原子加法與比較交換 (3.1) | 指令存在，但為 **Stub (直通)** | `kAtomicAdd` 與 `kAtomicCompSwap` 未實作並行執行緒的快取鎖定與互斥存取，無法執行無鎖資料結構。 |
| **數學與三角函數集** | 完整三角函數（`tan`, `asin`, `acos`, `atan`）與雙曲函數；浮點判定（`isnan`, `isinf`）；矩陣乘/轉置/行列式/逆矩陣 | 僅支援加減乘、Mad、倒數、開方倒數、Log2、Exp2、Sine、Cosine | 缺乏矩陣內置 ALU 支持，許多高級數學運算必須由編譯器展開為大量基本 ALU 指令，增加程式長度。 |
| **插值限定符 (Interpolation)** | 支援平滑插值（`smooth`）、平面不插值（`flat`）、中心插值（`centroid`）| 僅支援硬性 `FITRP` 流程 | 缺少 flat/centroid 插值限定符對應的平面方程式係數選擇，無法避免多重採樣時邊緣顏色的越界失真。 |
| **精確度限定符 (Precision)** | 支援並強制 `highp` (32b F), `mediump` (16b F), `lowp` (10b Fixed) 限定 | 全部以 **32-bit Float** 直通執行 | 忽略精確度限定符，無法模擬 mediump/lowp 對暫存器頻寬與 ALU 執行週期的節省效果。 |
| **Uniform 緩衝區 (UBO)** | 支援 Uniform Blocks 綁定至統一緩衝區對象（UBO） | 僅支援一組 flat shared registers | 無法載入大型 UBO 常數塊。 |

---

## 3. 開發任務矩陣 (To-Do List)

### 1. 偏導數運算與像素方塊協作 (Derivatives)
* [ ] **任務 1.1: 偏導數實作 (dFdx / dFdy)**
  * **細節**：在 `ExecuteFragmentPco` 中，`kDerivativeX` 應讀取 2x2 Quad 中同一列（horizontal neighbor）兩個 active lanes 的數值差；`kDerivativeY` 讀取同一行（vertical neighbor）的差值，而非 Stub 複製。
  * **修改檔**：[`pco_iss.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple%20CloudDocs/Codex/PvrGPU/src/systemc/shader/pco_iss.cpp)

### 2. 計算著色器與記憶體原子操作 (Atomics & Buffers)
* [ ] **任務 2.1: SSBO 記憶體載入/寫入實作**
  * **細節**：將 `kBufferLoad` 與 `kBufferStore` 連接至 MCU / SLC 記憶體交易請求，根據暫存器中的 GPU 位址進行讀寫。
  * **修改檔**：[`pco_iss.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple%20CloudDocs/Codex/PvrGPU/src/systemc/shader/pco_iss.cpp)、[`usc_cluster.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple%20CloudDocs/Codex/PvrGPU/src/systemc/shader/usc_cluster.cpp)
* [ ] **任務 2.2: 快取原子操作線路 (Cache-coherency Atomics)**
  * **細節**：實作 `kAtomicAdd` 與 `kAtomicCompSwap` 的記憶體互斥鎖定鎖定（Locking/Reservation）邏輯，在 SLC 或 L2 記憶體控制器中排隊原子指令以確保多執行緒執行安全。
  * **修改檔**：[`pco_iss.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple%20CloudDocs/Codex/PvrGPU/src/systemc/shader/pco_iss.cpp)

### 3. GLSL ES 3.x 數學函數與精確度限制 (Math & Precision)
* [ ] **任務 3.1: 高級三角函數與矩陣運算擴充**
  * **細節**：新增 `kFloatTangent`、`kFloatInverseTangent` 等 Opcode 的模擬實現；支援硬體輔助的矩陣快速轉置與乘法。
  * **修改檔**：[`pco_iss.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple%20CloudDocs/Codex/PvrGPU/src/systemc/shader/pco_iss.cpp)、[`pco_iss.h`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple%20CloudDocs/Codex/PvrGPU/src/systemc/shader/pco_iss.h)
* [ ] **任務 3.2: 16-bit Half-Float ALU 精度模擬 (mediump)**
  * **細節**：當指令被標記為 mediump 精確度時，在 ALU 運算前後對浮點數進行 16-bit 截斷（5-bit exp, 10-bit mantissa），以正確重現低精度浮點數在運算時的捨入誤差與 overflow 狀態。
  * **修改檔**：[`pco_iss.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple%20CloudDocs/Codex/PvrGPU/src/systemc/shader/pco_iss.cpp)

### 4. 插值限定符與 Uniform Blocks (Interpolation & UBOs)
* [ ] **任務 4.1: flat / centroid 插值模式支援**
  * **細節**：擴充 `FITRP` 指令，支援 flat 模式（直接複製 Provoking Vertex 頂點顏色值而不進行重心插值）與 centroid 模式（當採樣點超出多邊形時，限制採樣位置於多邊形邊界內）。
  * **修改檔**：[`pco_iss.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple%20CloudDocs/Codex/PvrGPU/src/systemc/shader/pco_iss.cpp)
* [ ] **任務 4.2: UBO (Uniform Buffer Object) 載入路徑**
  * **細節**：新增 `kConstantBufferLoad` 指令，允許 USC 透過常數快取（Constant Cache）異步從記憶體中分批載入 UBO 的資料塊至暫存器。
  * **修改檔**：[`pco_iss.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple%20CloudDocs/Codex/PvrGPU/src/systemc/shader/pco_iss.cpp)、[`usc_cluster.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple%20CloudDocs/Codex/PvrGPU/src/systemc/shader/usc_cluster.cpp)

---

## 4. 漸進式實作開發階段建議

1. **第一階段（偏導數與插值補齊）**：實作真實的 2x2 Quad 偏導數計算（dFdx/dFdy）；支援 flat 頂點插值模式以滿足 flat-shading 渲染。
2. **第一.五階段（數學庫擴展）**：新增其餘的三角函數支持（如 Tan），以及 mediump（16b 浮點）的局部運算截斷模擬。
3. **第二階段（記憶體對象與 UBO 支援）**：實作常數緩衝區讀取（UBO），解決 flat uniform 暫存器不足的問題。
4. **第三階段（計算著色器指令實作）**：實作真正的 SSBO Buffer Load/Store 與快取 atomic 操作，以完整支援 OpenGL ES 3.1+ Compute Shader。
