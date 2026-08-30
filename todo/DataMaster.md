# Data Master 與 PbeWriteBack — 模組分析與 OpenGL ES 3.x 補齊規劃

## 1. 架構概述

依照 PowerVR DXTP 的硬體參考架構，`data_master` 目錄下的模組屬於**資料輸入擷取側（Fetch Side）**，負責從 SLC / DRAM 記憶體子系統中讀取輸入數據（如頂點屬性、索引、常數、Blit 來源等），包裝並經由 PDS (Programmable Data Sequencer) 調度後送入 USC (Unified Shading Cluster) 等處理單元。

原先誤放於 `data_master/` 的 Framebuffer 寫出邏輯，已被遷移並重構為 [`fragment/PbeWriteBack`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple%20CloudDocs/Codex/PvrGPU/src/systemc/fragment/pbe_write_back.h) 模組，作為片段處理管線的最後寫出端。

### 五個 Data Master 與 PbeWriteBack 現狀

| 子模組 | 角色分類 | 有 SystemC SC_THREAD？ | 功能與對照描述 |
|--------|---------|-----------------------|---------------|
| `VertexDataMaster` | 📥 Input Fetch | ❌ (Stub) | 從 SLC 取頂點 (VBO/IBO) 資料送入 PDS |
| `PixelDataMaster` | 📥 Input Fetch | ❌ (Stub) | 從 SLC 取 pixel tile (深度/模板) 資料送入 PDS/ISP |
| `ComputeDataMaster`| 📥 Input Fetch | ❌ (Stub) | 從 SLC 取 compute data (SSBO/Uniforms) 送入 PDS |
| `DomainDataMaster` | 📥 Input Fetch | ❌ (Stub) | 從 SLC 取 Tessellation patch 資料送入 PDS |
| `TwoDDataMaster`   | 📥 Input Fetch | ❌ (Stub) | 從 SLC 取 2D blit src surface 送入 2D Blit Engine |
| `PbeWriteBack`     | 📤 Write Back | ✅ `Run()` 完整實作 | 片段管線最終寫出端：PBE FB -> PbeWriteBack -> SLC |

---

## 2. 核心資料型別

```cpp
struct PipelineTxn {
  PoolHandle state;       // MemoryPool 內的 PipelineState handle
  uint32_t   frame;
  uint64_t   sequence;
};

struct MemoryTxn {
  PipelineTxn         pipeline;        // 來源 pipeline txn
  PoolHandle          payload;         // 指向 MemoryPool 中的實體數據 (bulk bytes)
  uint64_t            address;         // GPU 虛擬位址
  uint64_t            bytes;           // payload 長度 (bytes)
  uint64_t            request_id;      // 單調遞增的請求識別 ID
  MemoryOperation     operation;       // kRead (讀取) / kWrite (寫入)
  MemoryClient        client;          // kFramebuffer / kMixedCache / kTextureCache / kUscL2
  MemoryPayloadFormat payload_format;  // kLinearBytes / kCacheLineWrites
};
```

---

## 3. PbeWriteBack — 偽碼（Pseudo Code）

```
// 常量與延遲定義
kFramebufferGpuAddress = 0x10000000
kPbeWriteBackLatency = 1 cycle

Process PbeWriteBack::Run():
  loop forever:
    // ① 接收來自 PBE (Pixel Back End) 的寫入通知
    txn = input.read()                      // blocking FIFO read (PipelineTxn)

    // ② 從 MemoryPool 載入 PipelineState 並驗證狀態
    state = LoadPipelineState(pool, txn.state)
    assert state.stage == kPbeComplete
    assert state.pbe_framebuffer is valid handle

    // ③ 驗證 Framebuffer 大小一致性 (固定 RGBA8)
    expected_bytes = state.width * state.height * 4
    assert expected_bytes != 0
    assert state.framebuffer_bytes == expected_bytes
    assert pool.Read(state.pbe_framebuffer).size() == expected_bytes

    // ④ 更新計數器與階段標記
    state.framebuffer_gpu_address = kFramebufferGpuAddress
    state.counters.pixel_data_master_transactions = 1
    state.counters.pixel_data_master_bytes = expected_bytes
    state.counters.pixel_data_master_cycles = kPbeWriteBackLatency
    state.counters.renderer_cycles += kPbeWriteBackLatency
    state.stage = kPixelDataMasterComplete

    // ⑤ 模擬延遲並將 PipelineState 回存
    WaitForCycles(kPbeWriteBackLatency)
    StorePipelineState(pool, txn.state, state)

    // ⑥ 組裝 MemoryTxn 並發出至 SLC (System Level Cache)
    memory_txn = {
      pipeline       = txn,
      payload        = state.pbe_framebuffer,
      address        = kFramebufferGpuAddress,
      bytes          = expected_bytes,
      operation      = kWrite,
      client         = kFramebuffer,
      payload_format = kLinearBytes
    }
    output.write(memory_txn)
```

---

## 4. 五個 Data Master — 偽碼（目前為 Stub 預留點）

```
// VertexDataMaster
Process VertexDataMaster::Run():
  // TODO: 自 SLC 預載入屬性與索引，包裝為 PDS Vertex task
  pass

// PixelDataMaster
Process PixelDataMaster::Run():
  // TODO: 自 SLC 預載入當前 Tile 的深度/模板數據
  pass

// ComputeDataMaster
Process ComputeDataMaster::Run():
  // TODO: 自 SLC 預載入 SSBO/Uniform 常數送至 PDS/USC
  pass

// DomainDataMaster
Process DomainDataMaster::Run():
  // TODO: 自 SLC 載入 Tessellation patch 控制頂點資料
  pass

// TwoDDataMaster
Process TwoDDataMaster::Run():
  // TODO: 自 SLC 載入 2D Blit 的 source pixel 數據
  pass
```

---

## 5. 與 OpenGL ES 3.x 功能缺失對比

### 5.1 PbeWriteBack（片段寫入端）

| OpenGL ES 3.x 功能 | 目前狀態 | 缺失與補齊路徑 |
|-------------------|---------|---------------|
| 多重渲染目標 (MRT) | ❌ 不支援 | 目前僅發出單一 `pbe_framebuffer`，需實作陣列或多個 output handles 分發 |
| 多樣化像素格式 | ❌ 硬編碼 RGBA8 | 須依 `state.pixel_format` 計算實際 BPP (如 RGBA16F, R11G11B10F 等) |
| Framebuffer 壓縮 (IMGIC) | ❌ 不支援 | 在送往 SLC 前需對接 `ImageCompression` 模組，進行 block 壓縮模擬 |

### 5.2 五大 Data Master（輸入擷取端）

| 模組 | OpenGL ES 3.x 功能缺失 | 補齊規劃 |
|-----|----------------------|---------|
| **VertexDM** | Instanced Rendering 頂點擷取頻率控制 | 依據 `AttribDivisor` 計算每實例累加 offset，向 SLC 請求數據 |
| **PixelDM** | 深度與模板的快速清除值 (Fast Clear) 與回讀 | 從記憶體讀取原有深度圖，用於 Z-Prepass 或遮擋遮罩比對 |
| **ComputeDM** | SSBO 隨機讀寫與原子緩衝區 (Atomic Buffers) | 實作雙向交易，封裝帶 `request_id` 的 `kRead`/`kWrite` 記憶體事務 |
| **TwoDDM** | `glBlitFramebuffer` 的紋理重採樣與拷貝 | 實作從 src 載入、格式轉換、縮放過濾後寫回 dst 的完整 DMA 流水線 |
| **DomainDM** | Tessellation Patch 資訊擷取 | 支援多個 patch 的控制點與常數定序加載 |

---

## 6. 關聯模組資料流圖

```
[頂點階段 (Geometry)]
  DRAM / SLC ──► VertexDataMaster (Fetch) ──► VertexPdsEngine ──► USC VS (著色)

[片段階段 (Fragment)]
  DRAM / SLC ──► PixelDataMaster (Fetch)  ──► PdsEngine ──► USC FS (著色)
                                                                 │
                                                                 ▼
  DRAM / SLC ◄── PbeWriteBack (Write-back) ◄── PBE (Pixel Back End) ◄┘
```
