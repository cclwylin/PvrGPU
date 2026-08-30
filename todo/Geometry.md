# PvrGPU 幾何管線設計與 OpenGL ES 3.x 補齊規劃

本文件詳細分析 PvrGPU 內部的幾何前端管線（VDM -> VertexFetch -> VertexPdsEngine -> USC VS -> ClipCull -> Tiler -> ParameterBuffer），並說明如何不改動 SystemC 模組框架下，進行 OpenGL ES 3.x 幾何特性擴充的實作細節。

---

## 1. PvrGPU 幾何前端管線架構

PvrGPU 的幾何階段模擬了 PowerVR TBDR 架構的幾何前端部分。

* **視覺化架構圖**：本目錄下提供了可編輯的 Draw.io 流程圖檔案：[`Geometry_pipeline.drawio`](<file:///Users/linwanyi/Library/Mobile%20Documents/com~apple~CloudDocs/Codex/PvrGPU/todo/Geometry_pipeline.drawio>)。您可直接將其匯入至 [draw.io](https://draw.io/) 進行編輯或瀏覽更美觀的彩色向量圖。
* **幾何管線資料流拓撲如下所示**：

```
[ Draw Call ] 
      |
      v
+-----------+      +-------------+      +--------------+      +-------------+
|    VDM    | ---> | VertexFetch | ---> |  Vertex PDS  | ---> |   USC VS    |
+-----------+      +-------------+      +--------------+      +-------------+
                                                                     | (VTXOUT)
                                                                     v
+-----------+      +-------------+      +-------------+
| Parameter | <--- |    Tiler    | <--- |  ClipCull   |
|  Buffer   |      +-------------+      +-------------+
+-----------+
```

### 1.1 五大核心幾何模組職責

1. **VDM (Vertex Data Master)**

   * **功能**：頂點資料管理器。接收幾何繪圖命令，驗證 VBO（頂點緩衝區）與 IBO（索引緩衝區）的邊界安全，並初始化輸入裝配（IA）的頂點/圖元計數器。
   * **本機 Patch 實作**：已擴充支援 `uint8_t`、`uint16_t` 與 `uint32_t` 三種索引格式的安全載入與解析；支援根據圖元繞向與 `Primitive Restart` 機制計算精確的 `ia_vertices` 與 `ia_primitives` 計數。
2. **VertexFetch (頂點擷取)**

   * **功能**：依據 `VertexAttributeBinding` 屬性繫結，從 VBO 記憶體中讀取頂點屬性，轉換為 32-bit float bits 載入至 PCO VTXIN 暫存器。
   * **快取與擴充實作**：除了 Direct-Mapped Post-Transform Cache 頂點重用快取外，本機 Patch 已實作幾何基元展開（Primitive Assembly Expansion）。在頂點著色執行前，將任意輸入拓撲（如 `TriangleStrip`/`TriangleFan`）以及 `Primitive Restart` 重啟分界線，轉譯並展開為標準的獨立 triangles 序列（`TriangleList`），藉此讓後續的 `ClipCull` 與 `Tiler` 模組無需任何修改即可完美支援多樣化圖元渲染。同時已實作多格式頂點屬性載入（包含 `Int8`/`Uint8`/`Int16`/`Uint16`/`Int32`/`Uint32`/`HalfFloat` 到 `Float32` 的 Normalized 歸一化轉換與 Raw Integer 直通讀取）。
3. **ClipCull (裁剪與剔除)**

   * **功能**：讀取頂點著色器輸出的 VTXOUT 暫存器（其中前 4 個為 Homogeneous 裁剪座標 $[x, y, z, w]^T$）。
   * **剪裁與剔除**：採用 Sutherland-Hodgman homogeneous 6 平面裁剪演算法。超出邊界的三角形被裁剪並透過三角形扇（Triangle Fan）重新裝配，並進行背面剔除（Face Culling，預設 CCW 繞向為正面），隨後投影轉換至 Viewport 窗口空間，並標準化繞向為逆時針。
4. **Tiler (分箱器)**

   * **功能**：光柵化前的空間劃分。將視埠內可光柵化的三角形按其 Bounding Box 與 32x32 尺寸的 Tile 進行相交測試，分發至對應 Tile 的 `TileRecord` 鏈結串列中。
5. **ParameterBuffer (參數緩衝區)**

   * **功能**：為後續的 ISP 光柵化與 USC 像素著色準備插值參數。
   * **係數計算**：量化為 24.8 定點數精度的 Edge Equations；針對 $1/W$ 以及各個 Varying 屬性除以 $W$ ($V/W$) 計算對應的平面插值係數（Plane Coefficients $A, B, C$）。

---

## 2. 幾何前端管線 Pseudo-code

```python
# 核心幾何管線流程虛擬碼

def execute_geometry_pipeline(state: PipelineState):
    # 1. VDM (頂點裝配驗證)
    validate_draw_command(state.draw)
    validate_vbo_limits(state.vertex_buffers)
  
    if is_indexed(state.draw):
        for idx in state.vertex_indices:
            assert idx < get_vertex_capacity(state.vertex_buffers)
        state.counters.ia_vertices = len(state.vertex_indices)
        state.counters.ia_primitives = len(state.vertex_indices) / 3
    else:
        state.counters.ia_vertices = state.draw.vertex_count
        state.counters.ia_primitives = 2  # 例如 FillSolid 滿版 Quad 為 2 個三角形

    # 2. VertexFetch (頂點擷取、Cache 模擬與幾何展開)
    lanes = []
    lane_refs = []
    pt_cache = Cache(slots=16) # Post-Transform Cache
  
    if not is_indexed(state.draw):
        for v_idx in range(state.draw.vertex_count):
            lane = fetch_attributes(state.vertex_buffers, v_idx)
            lanes.append(lane)
    else:
        # [Patch 新增] 拓撲與 Primitive Restart 展開
        # 將 TriangleStrip/TriangleFan 與 Restart Index 展開為標準 TriangleList
        expanded_indices = expand_topology_and_restart(
            state.vertex_indices, 
            state.draw.topology, 
            state.primitive_restart_enable, 
            state.primitive_restart_index
        )
    
        # 替換為展開後的 index 序列供後續管線使用
        state.vertex_indices = store_new_array(expanded_indices)
        state.draw.topology = PrimitiveTopology.kTriangleList
        state.draw.index_count = len(expanded_indices)
    
        for occurrence, v_idx in enumerate(expanded_indices):
            cache_slot = v_idx % pt_cache.size()
            entry = pt_cache[cache_slot]
        
            if not entry.valid or entry.vertex_index != v_idx:
                lane = fetch_attributes(state.vertex_buffers, v_idx)
                entry.vertex_index = v_idx
                entry.lane_index = len(lanes)
                entry.valid = True
                lanes.append(lane)
            
            lane_refs.append(LaneRef(entry.lane_index, v_idx))
        
    state.vertex_lanes = lanes
    state.vertex_lane_refs = lane_refs

    # [中介步驟：USC 執行 Vertex Shader 計算，輸出寫入 VTXOUT]

    # 3. ClipCull ( homogeneous 裁剪與繞向剔除 )
    triangles = []
    for primitive_idx in range(state.counters.ia_primitives):
        v0, v1, v2 = get_primitive_vertices(state.vertex_lanes, primitive_idx)
    
        # Sutherland-Hodgman 6 平面裁剪
        clipped_polygon = clip_homogeneous_6_planes(v0, v1, v2)
        if len(clipped_polygon) < 3:
            continue
        
        # Triangle Fan 重新裝配與投影
        for fan in range(2, len(clipped_polygon)):
            t_v0 = clipped_polygon[0]
            t_v1 = clipped_polygon[fan - 1]
            t_v2 = clipped_polygon[fan]
        
            w_v0 = viewport_transform(t_v0, state.width, state.height)
            w_v1 = viewport_transform(t_v1, state.width, state.height)
            w_v2 = viewport_transform(t_v2, state.width, state.height)
        
            front_facing = is_front_facing(t_v0, t_v1, t_v2, winding=CCW)
            face_culled = is_culled(state.raster_state.face_cull, front_facing)
        
            # Winding 標準化 (逆時針面積為正)
            area = get_quantized_area(w_v0, w_v1, w_v2)
            if area < 0:
                w_v1, w_v2 = w_v2, w_v1
                area = -area
            
            triangle = RasterTriangle(
                vertices=[w_v0, w_v1, w_v2],
                rasterizable=(area > 0 and not face_culled),
                face_culled=face_culled
            )
            triangles.append(triangle)
    state.raster_triangles = triangles

    # 4. Tiler (32x32 分箱)
    tiles = init_tile_grid(state.width, state.height, tile_size=32)
    primitive_refs = []
  
    for tile in tiles:
        tile.first_primitive_ref = len(primitive_refs)
        for idx, tri in enumerate(state.raster_triangles):
            if not tri.rasterizable:
                continue
            if intersects(tri.bbox, tile.rect):
                primitive_refs.append(TilePrimitiveRef(parameter_index=idx))
        tile.primitive_ref_count = len(primitive_refs) - tile.first_primitive_ref
    
    state.tile_records = tiles
    state.tile_primitive_refs = primitive_refs

    # 5. ParameterBuffer (平面係數計算)
    parameters = []
    coefficients = []
    for tri in state.raster_triangles:
        param = ParameterTriangle(key=tri.key, rasterizable=tri.rasterizable)
        if not tri.rasterizable:
            parameters.append(param)
            continue
        
        # 建立定點 Edge 方程式
        param.edge = build_edge_equations(tri.vertices)
        param.signed_area = get_quantized_area(tri.vertices)
        param.bbox = get_clamped_bbox(tri.vertices, state.width, state.height)
    
        # 建立透視插值平面平面參數
        coefficients.append(build_plane(tri.vertices, [1/w0, 1/w1, 1/w2]))
        for varying in tri.varyings:
            coefficients.append(build_plane(tri.vertices, varying * 1/W))
        
        parameters.append(param)
    
    state.parameter_triangles = parameters
    state.parameter_coefficients = coefficients
```

---

## 3. 對比 OpenGL ES 3.x 實作現狀與未來擴充

經過本次幾何前端 Patch，管線已具備完善的 GLES 3.x 前端特性相容力，剩餘未實作的功能彙整如下：

### 3.1 VDM & Input Assembly

1. **拓撲支援 (Primitive Topologies)**：
   * 
   * **未來待補 (Pending)**：`kPoints`、`kLines`、`kLineStrip`、`kLineLoop` 與 `kTriangleFan`（僅 VDM 完成統計，VertexFetch 與後續幾何光柵化流程需補齊線/點分箱相交測試與 Param 生成）。
2. 

### 3.3 ClipCull & Raster Setup

1. **自訂裁剪平面 (User Clip Planes / `gl_ClipDistance`)**：
   * **待補 (Pending)**：homogeneous 裁剪步驟加入對 `gl_ClipDistance[n]` 距離的判定與裁剪插值。
2. **深度夾擊 (Depth Clamping / `GL_DEPTH_CLAMP`)**：
   * **待補 (Pending)**：不剪切超出近/遠平面的頂點，改為直接 clamp 其深度值。
3. **Provoking Vertex (Flat Shading)**：
   * **待補 (Pending)**：不插值 Varying 時，指定以第一個或最後一個頂點的屬性為準。

### 3.4 幾何前端新增階段 (GLES 3.0/3.2)

1. **Transform Feedback (變換回饋 / Stream Out)**：
   * **待補 (Pending)**：VS 著色後、裁剪前，將屬性非同步寫回 Buffer 記憶體中。
2. **Geometry & Tessellation Shaders (幾何與細分著色器)**：
   * **待補 (Pending)**：新增 GS/TCS/TES 著色器階段，支援動態增加/細分圖元。
