# PvrGPU Pipelines TODO List / Task Matrix

此目錄包含 PvrGPU 幾何前端與片段後端模組的功能擴充規劃與對照表。

## 🎯 任務目標
為了讓 PvrGPU 管線能相容 OpenGL ES 3.x 並通過 GFXBench / dEQP 等真實應用幾何與片段測試，建立以下功能補齊之開發排程。

---

## 🛠️ 開發任務矩陣

### 1. VDM & Input Assembly 擴充
- [ ] **任務 1.1: 點與線的拓撲支援 (Topologies)**
  - 支援 `GL_POINTS`、`GL_LINES`、`GL_LINE_STRIP`、`GL_LINE_LOOP` 與 `GL_TRIANGLE_FAN`
  - 修改檔：[`vdm.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/geometry/vdm.cpp)
- [ ] **任務 1.2: 索引類型擴建 (Index Types)**
  - 支援 `GL_UNSIGNED_BYTE` (uint8) 與 `GL_UNSIGNED_INT` (uint32) 索引格式
  - 修改檔：[`vdm.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/geometry/vdm.cpp)
- [ ] **任務 1.3: 基元重啟機制 (Primitive Restart)**
  - 當遇到重啟索引值（`0xFF/0xFFFF/0xFFFFFFFF`）時中斷目前 primitive strip 並重新開始
  - 修改檔：[`vdm.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/geometry/vdm.cpp)

### 2. Vertex Fetch 擴充
- [ ] **任務 2.1: 頂點屬性格式與 Normalized 支援**
  - 實作整數格式（Int8, Uint8, Int16, Uint16, Int32, Uint32）與 Half-Float 的讀取與轉換
  - 實作整數常態化（Normalization）映射至 `[-1.0, 1.0]` 或 `[0.0, 1.0]`
  - 修改檔：[`vertex_fetch.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/geometry/vertex_fetch.cpp)
- [ ] **任務 2.2: 實例化資料載入 (Vertex Attrib Divisor)**
  - 依據 `instance_divisor` 按實例（instance）而非頂點進行步進讀取
  - 注入 `gl_InstanceID` 暫存器至 USC 頂點執行緒中
  - 修改檔：[`vertex_fetch.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/geometry/vertex_fetch.cpp)

### 3. ClipCull & Rasterizer Setup 擴充
- [ ] **任務 3.1: 使用者自訂剪裁面 (User Clip Planes)**
  - 在 homogeneous 裁剪步驟中加入對 `gl_ClipDistance[n]` 距離的判定與裁剪插值
  - 修改檔：[`clip_cull.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/geometry/clip_cull.cpp)
- [ ] **任務 3.2: 深度夾擊支援 (Depth Clamp)**
  - 當啟用 `GL_DEPTH_CLAMP` 時，關閉 Near/Far 裁剪面切除，並將 $z$ 投影 clamp 在 $[0, 1]$ 區間
  - 修改檔：[`clip_cull.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/geometry/clip_cull.cpp)
- [ ] **任務 3.3: Provoking Vertex (Flat Shading)**
  - 依據 flat shading 的設定選擇第一個或最後一個頂點屬性，作為整面三角形的插值色彩
  - 修改檔：[`clip_cull.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/geometry/clip_cull.cpp)

### 4. Tiler & Parameter Buffer 擴充
- [ ] **任務 4.1: 點與線的 Tiling 相交計算**
  - 支援線段相交測試（Line-Tile intersection）與考慮線寬
  - 支援點的相交測試，依 `gl_PointSize` 動態計算其覆蓋 Tile 的範圍
  - 修改檔：[`tiler.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/geometry/tiler.cpp)
- [ ] **任務 4.2: 點的 `gl_PointCoord` 係數產生**
  - 針對點圖元產生紋理座標平面方程式
  - 修改檔：[`parameter_buffer.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/geometry/parameter_buffer.cpp)

### 5. Transform Feedback 階段 (GLES 3.0)
- [ ] **任務 5.1: Transform Feedback 暫存器寫入**
  - 在頂點著色結束後、裁剪前，將指定的 VTXOUT 快取以 Stream Out 方式非同步寫回主記憶體中指定 buffer
  - 修改檔：幾何前端新增 Transform Feedback 控管模組或併入 [`clip_cull.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/geometry/clip_cull.cpp)

### 6. ISP & Depth/Stencil 擴充
- [ ] **任務 6.1: Late-Z / 延遲深度寫入機制**
  - 支援當 Fragment Shader 中存在 `discard`、`gl_FragDepth` 寫入或 `gl_SampleMask` 時的延遲深度測試與寫入，避免當前 ISP 的 Runtime Error 中斷
  - 修改檔：[`isp.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/fragment/isp.cpp)
- [ ] **任務 6.2: Stencil 模板測試與 Buffer 整合**
  - 在 `RasterState` 新增模板狀態，於 ISP 流程中進行 Stencil 比對與 Buffer 狀態更新
  - 修改檔：[`isp.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/fragment/isp.cpp)、[`functional_types.h`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/common/functional_types.h)
- [ ] **任務 6.3: 遮擋查詢 (Occlusion Queries)**
  - 統計通過深度/模板測試的可見片段，並將結果寫入查詢緩衝區中
  - 修改檔：[`isp.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/fragment/isp.cpp)

### 7. Fragment Frontend (Quad Assembly) 擴充
- [ ] **任務 7.1: MSAA 多重採樣與 Centroid 插值**
  - 擴充 coverage mask 支援多採樣，實作 Centroid 插值以防止採樣點越界，支援 Sample Shading 與 Resolve
  - 修改檔：[`fragment_frontend.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/fragment/fragment_frontend.cpp)

### 8. PBE & Blending 擴充
- [ ] **任務 8.1: 多渲染目標 (MRT) 支援**
  - 支援同時寫入多個 Render Target（`state.raster_state.draw_buffers`），並輸出多個 framebuffer 緩衝區
  - 修改檔：[`pbe.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/fragment/pbe.cpp)
- [ ] **任務 8.2: 進階混合模式與因子**
  - 實作等式（`SUBTRACT`、`REVERSE_SUBTRACT`、`MIN`、`MAX`）以及進階因子（`GL_CONSTANT_COLOR`、`GL_SRC_ALPHA_SATURATE` 等）
  - 修改檔：[`pbe.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple%20CloudDocs/Codex/PvrGPU/src/systemc/fragment/pbe.cpp)
- [ ] **任務 8.3: 多格式與 sRGB Framebuffer**
  - 支援 Half/Full Float, Integer 格式，並實作 sRGB 伽碼轉換
  - 修改檔：[`pbe.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple%20CloudDocs/Codex/PvrGPU/src/systemc/fragment/pbe.cpp)
- [ ] **任務 8.4: 顏色通道寫入遮罩 (Color Mask)**
  - 根據 `glColorMask` 設定，允許選擇性地寫入 R, G, B, A 色彩分量
  - 修改檔：[`pbe.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple%20CloudDocs/Codex/PvrGPU/src/systemc/fragment/pbe.cpp)

### 9. Data Master & PDS 擴充 (Fetch-Side & Schedule)
- [ ] **任務 9.1: Data Master Fetch 邏輯實作 (Fetch Path)**
  - 實作 VertexDataMaster、ComputeDataMaster、DomainDataMaster 和 TwoDDataMaster 的實體記憶體讀取與 fetch 協議，取代當前的 stub 預留點。
  - 修改檔：[`vertex_data_master.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple%20CloudDocs/Codex/PvrGPU/src/systemc/data_master/vertex_data_master.cpp) 等
- [ ] **任務 9.2: Vertex PDS Task Descriptor 處理**
  - 實作頂點階段 PDS task 產生與排程，驅動頂點 USC VS 執行緒的分配。
  - 修改檔：[`vertex_pds_engine.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple%20CloudDocs/Codex/PvrGPU/src/systemc/pds/vertex_pds_engine.cpp)

---

## 📈 開發階段建議

1. **第一階段（P0：幾何基礎與基礎混合）**：支援多樣化幾何 index、拓撲與 Primitive Restart；在片段後端補齊進階 Blend Equation/Factor 與 Color Mask 支援。
2. **第二階段（P1：屬性常態化、MRT 與 Stencil）**：解鎖頂點型態轉換與 Instancing；實作 PBE 多渲染目標 (MRT) 輸出，以及 ISP Stencil 測試迴路；對接 Vertex PDS Engine Task 定序器。
3. **第三階段（P2：高級裁剪、Late-Z、MSAA 與 Feedback）**：加入 Transform Feedback 與 Depth Clamp；實作 Late-Z 以處理 Shader discard/gl_FragDepth 寫入；補齊 MSAA 採樣器與 Resolve；實作完整 Data Master 的 Fetch-Side 通訊協定。
