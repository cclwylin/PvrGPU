# PvrGPU 紋理管線設計與 OpenGL ES 3.x 補齊規劃

本文件詳細分析 PvrGPU 內部的紋理採樣管線（TextureUnit），與 OpenGL ES 3.x 標準規格進行對比，找出功能性缺失與待補齊項目，並建立對應的開發任務矩陣。

---

## 1. PvrGPU 紋理採樣器架構與流程

PvrGPU 的 `TextureUnit` 模組模擬了 PowerVR 紋理處理單元（TPU）的硬體行為。

### 1.1 紋理採樣資料流
```
[ USC Shader kTextureSample ] 
          |
          v (PipelineStage::kFragmentTexturePending)
+------------------------------------+
|            TextureUnit             |
+------------------------------------+
|  1. Decode Rogue Image/Sampler     |
|     Descriptors                    |
|  2. Calculate Implicit LOD (Quads)  |
|  3. Issue Cache requests to TCU    |
|  4. Nearest/Bilinear/Trilinear     |
|     Filtering on Texels            |
+------------------------------------+
          |
          v (PipelineStage::kTextureComplete)
[ USC Shader WDF (Resume) ]
```

### 1.2 核心步驟與虛擬碼 (Pseudo-code)

以下為目前 `TextureUnit` 採樣處理的核心邏輯：

```python
# TextureUnit 紋理處理單元核心採樣流程虛擬碼

def run_texture_unit(state: PipelineState, pool: MemoryPool):
    # 驗證目前階段為紋理掛起狀態
    assert state.stage == PipelineStage.kFragmentTexturePending
    
    # 載入採樣請求、資源描述與常規暫存器
    requests = load_array(pool, state.texture_sample_requests)
    resources = load_array(pool, state.texture_resources)
    samplers = load_array(pool, state.sampler_states)
    shared = load_array(pool, state.fragment_shared_registers)
    
    resource = resources[0]
    
    # 1. 解碼 Rogue 硬體暫存器描述字 (Image 與 Sampler Word)
    image_words = shared[0:4]
    sampler_words = shared[8:12]
    image_desc = decode_rogue_image_descriptor(image_words)
    sampler_desc = decode_rogue_sampler_descriptor(sampler_words)
    
    # 2. 進行首次紋理分配至虛擬 DRAM (Texture Preload)
    if not texture_preloaded:
        preload_texture_to_dram(image_desc.gpu_address, resource.byte_size)
        texture_preloaded = True
        
    # 3. 導數與 LOD 計算 (隱式 Mipmap 選擇，以 2x2 Quad 為單位)
    implicit_lods = []
    if sampler_desc.mip_filter == TextureFilter.kLinear:
        for quad_idx in range(0, len(requests), 4):
            quad_coords = [requests[quad_idx + i].coordinates for i in range(4)]
            lod = compute_texture_implicit_lod(quad_coords, image_desc, sampler_desc)
            for _ in range(4):
                implicit_lods.append(lod)
                
    responses = []
    texel_fetch_count = 0
    
    # 4. 遍歷每個像素執行紋理採樣與過濾
    for idx, request in enumerate(requests):
        filtered_rgba = [0.0, 0.0, 0.0, 1.0]
        
        # 情況 A: Nearest 過濾 (無 Mipmap)
        if sampler_desc.min_filter == TextureFilter.kNearest and not mip_linear:
            u, v = nearest_repeat(request.u, image_desc.width), nearest_repeat(request.v, image_desc.height)
            texel = read_texel_from_cache(image_desc.gpu_address, level=0, u, v)
            filtered_rgba = [t / 255.0 for t in texel]
            texel_fetch_count += 1
            
        # 情況 B: Bilinear 雙線性過濾 (無 Mipmap)
        elif sampler_desc.min_filter == TextureFilter.kLinear and not mip_linear:
            x_axis = compute_linear_repeat(request.u, image_desc.width)
            y_axis = compute_linear_repeat(request.v, image_desc.height)
            
            # 讀取相鄰 4 個 texels (4 taps)
            t00 = read_texel_from_cache(image_desc.gpu_address, 0, x_axis.lower, y_axis.lower)
            t10 = read_texel_from_cache(image_desc.gpu_address, 0, x_axis.upper, y_axis.lower)
            t01 = read_texel_from_cache(image_desc.gpu_address, 0, x_axis.lower, y_axis.upper)
            t11 = read_texel_from_cache(image_desc.gpu_address, 0, x_axis.upper, y_axis.upper)
            
            # 使用雙線性插值 (Lerp)
            filtered_rgba = bilinear_interpolate(t00, t10, t01, t11, x_axis.weight, y_axis.weight)
            texel_fetch_count += 4
            
        # 情況 C: Trilinear 三線性過濾 (兩層 Mipmap bilinear 插值)
        else:
            lod = implicit_lods[idx]
            
            # 分別在 level0 與 level1 進行雙線性過濾
            lower_rgba = sample_bilinear(image_desc, lod.level0, request.u, request.v)
            upper_rgba = sample_bilinear(image_desc, lod.level1, request.u, request.v)
            
            # 進行層間線性插值
            filtered_rgba = lerp(lower_rgba, upper_rgba, lod.mip_weight)
            texel_fetch_count += 8
            
        responses.append(TextureSampleResponse(rgba=float_to_bits(filtered_rgba)))
        
    # 將過濾後 RGBA 寫回 MemoryPool
    state.texture_sample_responses = store_array(pool, responses)
    state.counters.texel_fetches += texel_fetch_count
    state.stage = PipelineStage.kTextureSamplesReady
    
    # 計算模擬時間並等待
    cycles = len(requests) * kReferenceUarch.texture_bypass_cycles
    wait_for_cycles(cycles)
    store_pipeline_state(pool, state)
```

---

## 2. 與 OpenGL ES 3.x 紋理功能對比分析

目前 PvrGPU 的紋理模擬僅限於滿足特定 GLBench case (`fill_tex_nearest` 與基本雙/三線性過濾) 的極簡子集，與完整 OpenGL ES 3.x 存在顯著功能缺失：

| 功能分類 | OpenGL ES 3.x 標準規格 | PvrGPU 目前狀態 | 關鍵缺失與硬體落差 |
| :--- | :--- | :--- | :--- |
| **紋理維度 (Dimensions)** | 2D, 3D, 2D Array, Cube Map, Cube Map Array, 2D Multisample | 僅支援 **2D** 紋理 | 缺少 3D 尋址、Array 索引選擇、Cube Map 的向量投影與無縫接縫過濾。 |
| **紋理格式 (Formats)** | RGBA8, RGB8, R8, RG8, Float32/16 家族, sRGB, ETC2/EAC, ASTC | 僅支援 **RGBA8 Unorm** | 缺少非 32bpp 格式支援、不支援浮點與整數直通紋理、缺少硬體解壓縮（ETC/ASTC）。 |
| **尋址模式 (Wrap Modes)** | `REPEAT`<br>`CLAMP_TO_EDGE`<br>`MIRRORED_REPEAT`<br>`CLAMP_TO_BORDER` | 僅支援 **`kRepeat`** | 尋址模式被硬編碼。缺少邊界裁減（Clamp）與鏡像尋址，這在非重置紋理邊緣會產生嚴重的渲染 artifacts。 |
| **過濾選項 (Filters)** | 任意 Min/Mag/Mip 組合 (例如 Nearest + Linear Mipmap Linear) | 僅支援 `LOD0 Nearest`<br>與 `Trilinear` 兩種硬性配置 | 拒絕其他 GLES 常見組合（例如 Bilinear 搭配 Nearest Mipmap，或 Mag=Linear But Min=Nearest 等）。 |
| **深度/陰影比對 (Depth Compare)**| 支援 Shadow samplers，能將深度紋理與 $R$ 值比對進行 PCF 陰影過濾 | 不支援比對機制 | 拒絕比對編碼，無法支援硬體陰影貼圖（Shadow Map）。 |
| **非等方性過濾 (Anisotropy)** | 支援最大 16x Anisotropic filtering | 不支援 (Bypass) | 拒絕相關硬體描述字設定，無法模擬斜角紋理拉伸時的細節清晰化。 |
| **多重採樣 (MSAA Texture)** | 支援 `sampler2DMS` 採樣多個 sub-samples | 不支援 | 無法進行 MSAA 紋理著色。 |
| **紋理重組 (Swizzle)** | 支援自訂 R, G, B, A 映射通道（如 `GL_TEXTURE_SWIZZLE_R`）| 僅支援恆等映射 (Identity) | 拒絕自訂 swizzle 遮罩，無法在硬體層完成通道快速對調。 |
| **紋理佈局 (Layout)** | 支援 Tiled / Twiddled 鋪磚式記憶體佈局以優化快取 | 僅支援 **`kLinear`** 佈局 | 硬體模擬未考慮 layout 轉換，對非 linear 紋理存取會產生亂序。 |

---

## 3. 開發任務矩陣 (To-Do List)

### 1. 尋址與邊界模式擴充 (Wrap Modes)
* [ ] **任務 1.1: 尋址模式描述解碼**
  * **細節**：解鎖 `DecodeRogueTextureSamplerDescriptor` 中對 U/V/W wrap 模式位元的嚴格 `!= 0U` 限制，將其解碼映射至 `TextureWrapMode` 列舉。
  * **修改檔**：[`texture_unit.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/texture/texture_unit.cpp)
* [ ] **任務 1.2: 邊緣夾擊與鏡像計算 (Clamp / Mirror)**
  * **細節**：修改 `ComputeTextureLinearRepeat` 與 `NearestRepeat`，當模式為 `CLAMP_TO_EDGE` 時進行 $[0, \text{extent}-1]$ 夾擊；為 `MIRRORED_REPEAT` 時根據奇偶區間進行對稱座標映射。
  * **修改檔**：[`texture_unit.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/texture/texture_unit.cpp)

### 2. 紋理格式與佈局擴充 (Formats & Layouts)
* [ ] **任務 2.1: 單/雙通道與整數/浮點格式支援**
  * **細節**：在 `read_texel` 中，根據解碼出的 `TextureFormat`，動態計算單個 texel 的位元組步進（如 R8 為 1B，RGBA16F 為 8B），並實作定點數/浮點數的解包，標準化輸出至過濾器。
  * **修改檔**：[`texture_unit.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/texture/texture_unit.cpp)、[`functional_types.h`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/common/functional_types.h)
* [ ] **任務 2.2: Tiled / Twiddled 座標轉換**
  * **細節**：若紋理佈局非 `kLinear`，在記憶體位移計算中加入 Morton 碼或對角鋪磚對照，以正確模擬硬體在快取局部性最佳化時的記憶體位址存取。
  * **修改檔**：[`texture_unit.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/texture/texture_unit.cpp)

### 3. 多維度與採樣器類型擴充 (Dimensions & Sampler Types)
* [ ] **任務 3.1: 3D 紋理與 Cube Map 尋址**
  * **細節**：擴充 `requests` 的尋址分量至 3D 坐標 $(r, g, b)$；實作 Cube map 的最大分量選擇與投影坐標歸一化，並支援無縫跨面（seamless face filtering）採樣。
  * **修改檔**：[`texture_unit.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/texture/texture_unit.cpp)、[`usc_cluster.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/shader/usc_cluster.cpp)
* [ ] **任務 3.2: 深度比對 (Shadow Samplers / PCF)**
  * **細節**：支援紋理狀態中 `GL_TEXTURE_COMPARE_MODE`。在採樣後，將插值的深度結果與參量 $R$ 進行比對運算（如 `LESS`、`GREATER`），輸出單一 $[0.0, 1.0]$ 的陰影強度。
  * **修改檔**：[`texture_unit.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/texture/texture_unit.cpp)

### 4. 任意過濾組合與 Anisotropic 支援
* [ ] **任務 4.1: 解除過濾模式配對鎖定**
  * **細節**：重構 `DecodeRogueTextureSamplerDescriptor`，使其允許任意 `min_filter` 與 `mip_filter` 組合（如 Bilinear + Mipmap Nearest），並在 `SampleRun` 迴圈中按規格靈活調度 1-tap, 4-taps 或多層採樣流程。
  * **修改檔**：[`texture_unit.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/texture/texture_unit.cpp)
* [ ] **任務 4.2: 仿射/非等方性過濾模擬**
  * **細節**：根據 Quad 偏導數計算非對稱足跡（footprint），進行多點等步長（lines of anisotropic taps）線性採樣並加權平均。
  * **修改檔**：[`texture_unit.cpp`](file:///Users/linwanyi/Library/Mobile%20Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/texture/texture_unit.cpp)

---

## 4. 漸進式實作開發階段建議

1. **第一階段（尋址與過濾自由化）**：解鎖 Sampler Word 的嚴格檢查；實作 U/V 座標的 `CLAMP_TO_EDGE`；允許 Nearest/Bilinear/Trilinear 自由搭配，確保與多種 GLES 渲染模式的尋址安全。
2. **第二階段（格式與維度擴展）**：新增 Cube Map 採樣與 2D Array 支援；新增 R8, RG8 以及 RGB565 格式的定點解包。
3. **第三階段（深度比對與效能精緻化）**：實作 PCF 深度陰影紋理比對；實作 Tiled Layout 記憶體尋址以完成真實的紋理快取命中率（TCU Hit Rate）校準。
