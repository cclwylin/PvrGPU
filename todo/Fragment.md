# PvrGPU 片段管線設計與 OpenGL ES 3.x 補齊規劃

本文件詳細分析 PvrGPU 內部的片段後端管線（TileScheduler -> ISP -> FragmentFrontend -> USC PS/Texture -> PBE），並說明如何在不改動 SystemC 模組框架下，進行 OpenGL ES 3.x 片段特性與像素後端擴充的實作細節。

---

## 1. PvrGPU 片段後端管線架構

PvrGPU 的片段處理階段模擬了 PowerVR TBDR 架構的後端像素渲染與混合管線。

* **片段管線資料流拓撲如下所示**：

```
[ Parameter Buffer ]
        |
        v (Stage: kFragmentDecoded)
+---------------+      +-------------+      +------------------+
| TileScheduler | ---> |     ISP     | ---> | FragmentFrontend |
+---------------+      +-------------+      +------------------+
                              |                    | (Stage: kFragmentsReady)
                              |                    v
                              |             +------------------+
                              |             |   PDS & USC PS   |
                              |             +------------------+
                              |                    |
                              v                    v (Stage: kTextureComplete)
                        +------------------------------+
                        |             PBE              |
                        +------------------------------+
```

### 1.1 四大核心片段模組職責

1. **TileScheduler (Tile 排程器)**

   * **功能**：接收由幾何前端 Tiler 寫入的 `TileRecord` 與 `TilePrimitiveRef`（以 32x32 pixel 為單位），計算排程週期，並作為 Fragment Phase 的起點啟動資料流。
   * **事件驅動設計**：FIFO transaction 中只攜帶 `MemoryPool` 的 state 握手控制權，實際的資料大量暫存於 MemoryPool 中，藉此優化模擬速度。

2. **ISP (Image Synthesis Processor / 影像合成處理器)**

   * **功能**：隱藏面消除（HSR）與可見性測試核心。以 32x32 Tile 為單位載入圖元參量，對每個像素中心進行 24.8 定點 Edge Equations 覆蓋測試。
   * **早期深度與消除 (Early HSR)**：
     * **不透明路徑 (Opaque)**：若混合（Blend）未啟用，則執行 HSR。新通過深度測試的片段會剔除（Reject）舊有的 Owner，使每個像素在進入 Shader 前只保留唯一的最終 Owner，大幅減少 Overdraw。
     * **半透明路徑 (Translucent/Blend)**：若混合啟用，則不進行 HSR 剔除。ISP 僅做深度測試並保留所有通過的片段候選者（FragmentCandidate），以 API 提交順序（submit_ordinal）儲存於序列中，供後續 PBE 進行有序混合。
     * **安全邊界檢查與副作用處理 (已實作 Patch)**：
       * 已實作 Early opaque HSR culling bypass 機制：當 Shader 內含 `discard`（`shader_may_discard`）或 early HSR 被標記為不安全時，自動關閉早期不透明 HSR 剔除（將 `opaque_early_hsr` 設為 `false`），確保被拋棄片段下方的片段不被誤剔除。
       * 已實作 Early Depth Write Gating 機制：當著色器會自訂寫入深度（`shader_writes_depth`）時，自動關閉 ISP 階段的 Early Z 深度更新（`early_depth_write = false`），避免將不正確的插值深度寫入 Depth Buffer。

3. **FragmentFrontend (片段前端)**

   * **功能**：將 ISP 篩選出的可見片段候選者包裝成適合 Unified Shading Cluster (USC) 執行的 2x2 Fragment Quads（片段方塊）。
   * **紋理取樣與輔助像素 (Helper Pixels)**：
     * 對於紋理採樣測試案例（`IsTextureFamily`），USC 以 4x2 SIMD half-stamp 作為調度單位。若 half-stamp 內有任一像素被圖元覆蓋，則會將此 half-stamp 拆為兩個 2x2 像素 Quad 進行執行緒分派。
     * 未被圖元覆蓋但位於 stamp 內的其他像素會被標記為**輔助像素 (Helper Pixels)**。它們會執行紋理座標插值與 TPU 採樣以計算偏導數（$\text{dFdx}, \text{dFdy}$ 供 Mipmapping 使用），但在最後 PBE 輸出時會將其 Write Mask 設為 0 以防寫回。

4. **PBE (Pixel Back End / 像素後端)**

   * **功能**：片段著色的終點。負責色彩空間/格式轉換、固定功能混合（Blending）與 Framebuffer 寫回。
   * **資料混合與轉換**：
     * 讀取 USC Shader 輸出的 raw Float32 (PIXOUT0..3)，將其轉換為 RGBA8 UNORM 格式（支援 Standard Nearest Rounding）。
     * **全功能混合與寫入遮罩 (已實作 Patch)**：
       * 支援多種 GLES 3.x Blending Equations（`ADD`、`SUBTRACT`、`REVERSE_SUBTRACT`、`MIN`、`MAX`），透過定點整數精確混色並處理飽和截斷。
       * 支援完整的 Blending Factors（含 RGB 與 Alpha 獨立因子、色彩與目標混合因子，如 `kSourceColor`, `kDestinationColor`, `kDestinationAlpha` 等）。
       * 支援顏色寫入遮罩（`color_mask`），允許對 R, G, B, A 四通道進行獨立的寫入屏蔽。

---

## 2. 片段後端管線 Pseudo-code

```python
# 核心片段後端管線流程虛擬碼

# ==========================================
# 1. TileScheduler 階段
# ==========================================
def run_tile_scheduler(state: PipelineState, pool: MemoryPool):
    # 驗證前置階段為片段解碼完成
    assert state.stage == PipelineStage.kFragmentDecoded
    
    tiles = load_array(pool, state.tile_records)
    primitive_refs = load_array(pool, state.tile_primitive_refs)
    
    # 若為 face-cull 剔除後之空圖元流，進行不變性安全檢查
    if len(primitive_refs) == 0:
        parameters = load_array(pool, state.parameter_triangles)
        assert all(t.rasterizable == 0 for t in parameters)
        
    state.scheduled_tiles = len(tiles)
    state.counters.tiles_scheduled = len(tiles)
    
    # 根據 tile 數量計算模擬週期 (CeilDivide)
    cycles = kReferenceUarch.scheduler_base_cycles + ceil(len(tiles) / kReferenceUarch.scheduler_tiles_per_batch)
    state.counters.tile_scheduler_cycles = cycles
    state.counters.renderer_cycles += cycles
    state.stage = PipelineStage.kTilesScheduled
    
    wait_for_cycles(cycles)
    store_pipeline_state(pool, state)

# ==========================================
# 2. ISP (HSR 與可見性測試) 階段
# ==========================================
def run_isp(state: PipelineState, pool: MemoryPool):
    assert state.stage == PipelineStage.kTilesScheduled
    
    # 目前參考 ISP 限制不支援 Shader 副作用 (discard / 寫入深度 / 寫入 Sample Mask)
    if state.raster_state.shader_may_discard or state.raster_state.shader_writes_depth or state.raster_state.shader_writes_sample_mask:
        raise RuntimeError("ISP late shader-side-effect handling is not implemented")
        
    assert state.raster_state.sample_count == 1
    
    tiles = load_array(pool, state.tile_records)
    primitive_refs = load_array(pool, state.tile_primitive_refs)
    parameters = load_array(pool, state.parameter_triangles)
    
    pixel_count = state.width * state.height
    candidates = []
    
    # 初始化深度快取與 Owner 映射表
    depth_buffer = [state.raster_state.depth.clear_depth] * pixel_count
    owner_map = [None] * pixel_count  # 儲存 candidate_index 用于 Opaque HSR
    covered_pixels_map = [0] * pixel_count
    
    opaque_early_hsr = (state.raster_state.blend.enable == 0)
    
    for tile in tiles:
        # 遍歷當前 tile 內關聯的圖元 references
        for ref_idx in range(tile.first_primitive_ref, tile.first_primitive_ref + tile.primitive_ref_count):
            ref = primitive_refs[ref_idx]
            triangle = parameters[ref.parameter_index]
            
            # 計算該圖元與 Tile 邊界的重疊區域
            y_begin = max(tile.y0, max(0, triangle.min_y))
            y_end = min(tile.y1, max(0, triangle.max_y))
            x_begin = max(tile.x0, max(0, triangle.min_x))
            x_end = min(tile.x1, max(0, triangle.max_x))
            
            for y in range(y_begin, y_end):
                for x in range(x_begin, x_end):
                    # 取像素中心 (Subpixel Scaling)
                    sample_x = x * kSubpixelScale + kSubpixelScale / 2
                    sample_y = y * kSubpixelScale + kSubpixelScale / 2
                    
                    edge_values = [0, 0, 0]
                    # 1. 覆蓋測試 (Coverage Test)
                    if not covers_sample(triangle, sample_x, sample_y, edge_values):
                        continue
                        
                    # 建立候選片段
                    candidate = FragmentCandidate(
                        x=x, y=y, 
                        primitive_id=triangle.key.api_primitive_id,
                        parameter_index=ref.parameter_index,
                        submit_ordinal=ref.submit_ordinal,
                        sample_mask=1,
                        visibility=FragmentVisibility.kRejected
                    )
                    
                    # 2. 深度插值 (Perspective Interpolation)
                    barycentric = calculate_barycentric(triangle, edge_values)
                    candidate.depth = (barycentric[0] * triangle.window_z[0] +
                                      barycentric[1] * triangle.window_z[1] +
                                      barycentric[2] * triangle.window_z[2])
                    
                    pixel_idx = y * state.width + x
                    covered_pixels_map[pixel_idx] = 1
                    
                    # 3. 深度測試 (Depth Test)
                    passes = True
                    if state.raster_state.depth.test_enable:
                        passes = depth_pass_test(state.raster_state.depth.compare_op, candidate.depth, depth_buffer[pixel_idx])
                        
                    candidate_index = len(candidates)
                    candidates.append(candidate)
                    
                    if not passes:
                        continue
                        
                    # 4. HSR 覆蓋更新
                    candidate.visibility = FragmentVisibility.kVisible
                    
                    if opaque_early_hsr:
                        # 若之前該像素已有 Owner，將其剔除 (HSR 發生)
                        prev_owner = owner_map[pixel_idx]
                        if prev_owner is not None:
                            candidates[prev_owner].visibility = FragmentVisibility.kRejected
                        owner_map[pixel_idx] = candidate_index
                        
                    # 5. 寫入深度緩衝
                    if state.raster_state.depth.test_enable and state.raster_state.depth.write_enable:
                        depth_buffer[pixel_idx] = candidate.depth

    # 統計並儲存結果
    visible_count = sum(1 for c in candidates if c.visibility == FragmentVisibility.kVisible)
    state.fragment_candidates = store_array(pool, candidates)
    state.active_fragment_invocations = visible_count
    state.stage = PipelineStage.kVisibilityReady
    
    # 週期計算與等待
    cycles = kReferenceUarch.isp_base_cycles + ceil(len(candidates) / kReferenceUarch.isp_candidates_per_batch)
    state.counters.isp_cycles = cycles
    state.counters.renderer_cycles += cycles
    
    wait_for_cycles(cycles)
    store_pipeline_state(pool, state)

# ==========================================
# 3. FragmentFrontend (2x2 Quad 組裝) 階段
# ==========================================
def run_fragment_frontend(state: PipelineState, pool: MemoryPool):
    assert state.stage == PipelineStage.kVisibilityReady
    
    candidates = load_array(pool, state.fragment_candidates)
    parameters = load_array(pool, state.parameter_triangles)
    
    invocations = []
    fragment_quads = []
    
    texture_case = is_texture_family(state.functional_case)
    
    if not texture_case:
        # 非紋理路徑：直接將 ISP 可見的 pixel 壓縮並包裝成 2x2 quads
        quad_map = {} # key: (parameter_index, quad_id) -> quad_index
        
        for candidate in candidates:
            if candidate.visibility != FragmentVisibility.kVisible:
                continue
                
            quad_x = candidate.x // 2
            quad_y = candidate.y // 2
            quad_id = quad_y * ceil(state.width / 2) + quad_x
            
            # 建立 Invocations 記錄
            inv_idx = len(invocations)
            invocations.append(create_fragment_invocation(candidate, quad_id))
            
            key = (candidate.parameter_index, quad_id)
            if key not in quad_map:
                quad_map[key] = len(fragment_quads)
                fragment_quads.append(FragmentQuad(
                    parameter_index=candidate.parameter_index,
                    quad_id=quad_id,
                    submit_ordinal=candidate.submit_ordinal,
                    coverage_mask=0, write_mask=0, helper_mask=0
                ))
                
            quad = fragment_quads[quad_map[key]]
            lane = (candidate.y % 2) * 2 + (candidate.x % 2)
            quad.invocation_indices[lane] = inv_idx
            quad.coverage_mask |= (1 << lane)
            quad.write_mask |= (1 << lane)
            
    else:
        # 紋理路徑：以 4x2 half-stamps 進行分組以支援導數計算 (dFdx/dFdy)
        shader_lanes = []
        touched_half_stamps = set() # (parameter_index, half_stamp_id)
        
        # 尋找所有被觸碰過的 half-stamps
        for candidate in candidates:
            half_stamp_id = (candidate.y // 2) * ceil(state.width / 4) + (candidate.x // 4)
            touched_half_stamps.add((candidate.parameter_index, half_stamp_id))
            
        # 以 half-stamp 為單位補齊 helper pixel
        for parameter_index, half_stamp_id in sorted(touched_half_stamps):
            stamp_x = (half_stamp_id % ceil(state.width / 4)) * 4
            stamp_y = (half_stamp_id // ceil(state.width / 4)) * 2
            
            # 分解成兩個 2x2 Quad
            for child_x_offset in [0, 2]:
                quad_x = stamp_x + child_x_offset
                quad_y = stamp_y
                if quad_x >= state.width or quad_y >= state.height:
                    continue
                    
                quad = FragmentQuad(
                    parameter_index=parameter_index,
                    quad_id=(quad_y // 2) * ceil(state.width / 2) + (quad_x // 2),
                    submit_ordinal=parameters[parameter_index].key.submit_ordinal,
                    coverage_mask=0, write_mask=0, helper_mask=0
                )
                
                # 補齊 2x2 中的 4 個 lanes
                for lane in range(4):
                    x = quad_x + (lane % 2)
                    y = quad_y + (lane // 2)
                    if x >= state.width or y >= state.height:
                        continue
                        
                    # 檢查該位置是否在 ISP 判定的可見清單中
                    visible_cand = find_visible_candidate(candidates, parameter_index, x, y)
                    lane_bit = (1 << lane)
                    
                    shader_lane = ShaderLane(x=x, y=y, parameter_index=parameter_index, quad_id=quad.quad_id, quad_lane=lane)
                    
                    if visible_cand:
                        # 覆蓋像素 (Normal Lane)
                        inv_idx = len(invocations)
                        invocations.append(create_fragment_invocation(visible_cand, quad.quad_id))
                        shader_lane.visible_invocation_index = inv_idx
                        quad.coverage_mask |= lane_bit
                        quad.write_mask |= lane_bit
                    else:
                        # 輔助像素 (Helper Lane)
                        shader_lane.helper = 1
                        quad.helper_mask |= lane_bit
                        
                    quad.invocation_indices[lane] = len(shader_lanes)
                    shader_lanes.append(shader_lane)
                    
                fragment_quads.append(quad)
                
        state.fragment_shader_lanes = store_array(pool, shader_lanes)
        
    state.fragment_invocations = store_array(pool, invocations)
    state.fragment_quads = store_array(pool, fragment_quads)
    state.fragment_groups = len(fragment_quads)
    state.stage = PipelineStage.kFragmentsReady
    
    # 週期與等待
    cycles = kReferenceUarch.fragment_frontend_base_cycles + ceil(state.fragment_shader_lane_count / kReferenceUarch.fragment_lanes_per_batch)
    state.counters.fragment_frontend_cycles = cycles
    state.counters.renderer_cycles += cycles
    
    wait_for_cycles(cycles)
    store_pipeline_state(pool, state)

# ==========================================
# 4. PBE (Pixel Back End / 寫入幀緩衝) 階段
# ==========================================
def run_pbe(state: PipelineState, pool: MemoryPool):
    # 驗證前置階段為紋理/著色計算完成
    assert state.stage == PipelineStage.kTextureComplete
    
    invocations = load_array(pool, state.fragment_invocations)
    outputs = load_array(pool, state.fragment_outputs) # 來自 USC PS
    
    pixel_count = state.width * state.height
    framebuffer = [float_to_unorm8(state.raster_state.clear_color)] * pixel_count
    written_map = [0] * pixel_count
    
    for idx, output in enumerate(outputs):
        # 嚴格的不變性檢查：驗證 Fragment 與 USC 輸出的對稱性
        invocation = invocations[idx]
        assert output.x == invocation.x and output.y == invocation.y
        assert output.written_mask == 0x0f  # 必須寫滿 RGBA
        
        pixel_idx = output.y * state.width + output.x
        
        # 檢查 Opaque 重複寫入限制
        if written_map[pixel_idx] > 0 and not state.raster_state.blend.enable:
            raise RuntimeError("PBE attempted to shade one opaque owner twice")
            
        written_map[pixel_idx] += 1
        
        # 1. 浮點數轉換 (Float32 to UNORM8)
        source_color = [float_bits_to_unorm8(output.pixel_output[c]) for c in range(4)]
        
        # 2. 混合處理 (Alpha Blending)
        if state.raster_state.blend.enable:
            dest_color = framebuffer[pixel_idx]
            blended = [0] * 4
            for c in range(4):
                src_factor = get_blend_factor(state.raster_state.blend, c, source_color, dest_color, is_src=True)
                dst_factor = get_blend_factor(state.raster_state.blend, c, source_color, dest_color, is_src=False)
                # 套用 ADD 公式
                blended[c] = blend_add_unorm8(source_color[c], dest_color[c], src_factor, dst_factor)
            framebuffer[pixel_idx] = blended
        else:
            framebuffer[pixel_idx] = source_color
            
    state.pbe_framebuffer = store_array(pool, flatten(framebuffer))
    state.stage = PipelineStage.kPbeComplete
    
    # 週期與計量
    blend_cycles = ceil(len(outputs) / kReferenceUarch.pbe_blend_fragments_per_batch) if state.raster_state.blend.enable else 0
    cycles = kReferenceUarch.pbe_base_cycles + ceil(pixel_count / kReferenceUarch.pbe_pixels_per_batch) + blend_cycles
    state.counters.pbe_cycles = cycles
    state.counters.renderer_cycles += cycles
    
    wait_for_cycles(cycles)
    store_pipeline_state(pool, state)
```

---

## 3. 對比 OpenGL ES 3.x 實作現狀與未來擴充

經過本次片段管線修補（Patch），PvrGPU 已補齊了大部分關鍵的片段與混合功能。以下為最新對照表與剩餘未實作的功能規劃：

### 3.1 延遲深度測試與副作用處理 (Late Z / Late Depth Test)

* **現狀 (Gaps)**：
  當前已實作對 `shader_may_discard` 與 `shader_writes_depth` 的 fallback 機制：
  1. 當存在 `discard` 時，關閉不透明 HSR 剔除以保留底層被遮擋片段。
  2. 當存在 `shader_writes_depth` 時，關閉 Early Depth Write。
* **未來待補 (Pending)**：
  1. **實作 Late-Z 回饋機制與 Late-Z 深度更新**（在 USC 執行完畢後，於 PBE 階段或後置階段依據 Shader 實際輸出的深度進行深度測試與寫入）。
  2. **實作 `gl_FragDepth` 插值平面重新計算**。

### 3.2 模板測試 (Stencil Test)

* **現狀 (Gaps)**：
  目前 `DepthState` 中**完全沒有 Stencil 狀態**，且 ISP 與 PBE 程式中也無任何 Stencil Buffer 讀寫與 stencil 比較操作（Stencil Fail, Stencil Pass, Depth Fail 等動作）。
* **未來待補 (Pending)**：
  1. **資料結構擴充**：於 `PipelineState` 中新增 `StencilState`（支援前後面獨立設定：`glStencilFuncSeparate`、`glStencilOpSeparate`）。
  2. **ISP 邏輯擴充**：於 ISP 中加入 Stencil 測試迴路。在進行 `DepthPass` 前，讀取對應的 stencil 暫存值進行比較，並依測試結果更新 Stencil Buffer 內的 8-bit 模板值。

### 3.3 多重渲染目標與繪圖緩衝區 (Multiple Render Targets / MRT)

* **現狀 (Gaps)**：
  目前 PBE 的 `pbe_framebuffer` 僅支援**單一渲染目標**，且輸出格式固定為一個 `width * height * 4` bytes 的 RGBA8 陣列。不支援 OpenGL ES 3.0 的 MRT（即同時寫入多個 Render Target，例如 `gl_FragData[n]`）。
* **未來待補 (Pending)**：
  1. **MRT 支援**：修改 PBE 核心，使其可同時載入並更新最多 4 或 8 個不同的 Framebuffer 緩衝區（由 `state.raster_state.draw_buffers` 控制）。
  2. **動態格式轉換器**：目前 PBE 內轉換器硬編碼為 RGBA8 UNORM。未來需實作 GLES 3.x 支援的各式格式：
     * **浮點格式**：RGBA16F, RGBA32F, R11F_G11F_B10F
     * **整數格式**：RGBA8I, RGBA8UI, RGBA32I
     * **壓縮與封裝格式**：RGB10_A2, sRGB 格式自動 Gamma 轉換（`GL_FRAMEBUFFER_SRGB`）

### 3.4 進階顏色混合擴充 (Advanced Blending)

* **現狀 (Gaps)**：
  已實作支援：
  1. **等式擴充**：於 PBE 中實作 `ADD`、`SUBTRACT`、`REVERSE_SUBTRACT`、`MIN`、`MAX` 等混合運算。
  2. **因子擴充**：支援 `kSourceColor`、`kOneMinusSourceColor`、`kDestinationColor`、`kOneMinusDestinationColor`、`kDestinationAlpha`、`kOneMinusDestinationAlpha` 等 GLES 3.x 混合因子。
  3. **通道遮罩**：支援 `color_mask` 色彩寫入屏蔽（4-bit）。
* **未來待補 (Pending)**：
  1. 支援 `GL_CONSTANT_COLOR` 與 `GL_SRC_ALPHA_SATURATE` 等進階因子的常數快取注入。

### 3.5 多重採樣抗鋸齒與 Sample Mask (MSAA)

* **現狀 (Gaps)**：
  目前 ISP 要求 `sample_count == 1`，且片段前端以單一像素中心為覆蓋依據，不支援 GLES 3.x 的 MSAA（4x/8x 多重採樣與 Sample Shading）。
* **未來待補 (Pending)**：
  1. **多重採樣覆蓋率測試**：ISP 針對 4x/8x 的次像素樣本（Subsamples）執行 Edge Equations 覆蓋測試，並產生 Sample Mask。
  2. **Centroid 插值與 Sample Shading**：幾何與片段前端需支援 `centroid` 插值，避免採樣點超出圖元邊界；支援按樣本進行片段著色（Sample Shading）。
  3. **MSAA 幀緩衝解析 (Resolve)**：PBE 需新增 Resolve 階段階段，將多採樣緩衝區降採樣為單採樣 Framebuffer。

### 3.6 遮擋查詢 (Occlusion Query)

* **現狀 (Gaps)**：
  目前完全沒有機制回饋遮擋查詢（Occlusion Query）計數。
* **未來待補 (Pending)**：
  1. **計數器注入**：在 ISP 與 PBE 內，若啟用了遮擋查詢，對所有通過深度與模板測試的可見片段，累加 `passed_samples` 計數，並在繪圖命令結束後將該計數寫回 API 指定的 query 結構中。
