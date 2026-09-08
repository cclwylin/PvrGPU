# dEQP 既有失敗修復：2026-09-08

本輪針對先前討論的 Tessellation、Geometry Shader、Basic Compute、SSBO
完成實作與除錯。四組使用同一版中央 **graphics API 27 / compute API 4**
完整重跑，不挑選子集合、不修改 stock CTS 判定，也不把 NotSupported 算成 Pass。

## 最終結果

| 完整群組 | 案例數 | Pass | Fail | NotSupported | NoResult |
| --- | ---: | ---: | ---: | ---: | ---: |
| `functional.tessellation.*` | 406 | 400 | 0 | 6 | 0 |
| `functional.geometry_shading.*` | 207 | 197 | 0 | 10 | 0 |
| `functional.compute.basic.*` | 41 | 41 | 0 | 0 | 0 |
| `functional.ssbo.*` | 2061 | 2060 | 0 | 1 | 0 |
| 合計 | 2715 | 2698 | 0 | 17 | 0 |

以上皆為 GLES31 stock CTS。最終執行時間為臺灣時間 2026-09-08
13:20–13:26，所有群組 runtime 前後 SHA-256 相同。沒有 timeout、
未驗證的 Pass、Pass 案例內的 unsupported draw／dispatch 或模型錯誤。

這是「四組既有 Fail 清零」，不是完整 GLES conformance，也不是所有能力與
llvmpipe 相同。保留的不支援項目為：

- Tessellation：6 個 ES 3.2 negative compile 案例的版本能力門檻。
  llvmpipe 基線為 406 Pass；PvrGPU 不以提高版本宣告來跳過實作要求。
- Geometry：1 個要求 2048 output components、1 個要求 point size 5，
  以及 8 個要求 multisample 2D-array extension 的案例。
  llvmpipe 基線為 206 Pass / 1 NotSupported，尚有 9 案能力差距。
- SSBO：`layout.random.all_per_block_buffers.45` 要求的 compute shader
  storage block 數量超過目前公開上限。

## 修正內容

- Tessellation：獨立 StreamOutput 模組直接捕獲真正 TES domain outputs；
  保留 TF-only exports 與明確 last-stage varying linkage。補齊原生 AND、
  register-allocation 使用 VTXIN 的合法範圍，以及常數 varying 在極小三角形
  的平面計算。沒有以 patch inputs 冒充 TES 輸出。
- Geometry：接通 layered color/depth/stencil 的 LOAD、逐層 raster/PBE、
  全 attachment readback，以及原 provoking vertex 的 Layer；明確處理 FS
  `gl_Layer` 的 flat raw-DWORD linkage。資源 mip/row/layer span 全部先驗界。
- GS 紋理：原生 SMP/WDF 經獨立 GeometryShader 與 TextureUnit FIFO 執行；
  補齊 GS TEX producer、DrawList、JSON 與 protocol 的一致性檢查。
- Cubemap resolve：以標準 NIR if-to-select 降低受限、uniform、無副作用的
  explicit-LOD 分支。未選取分支的座標／LOD 由原生 ALU 選用安全輸入，
  被選取分支仍保留應用程式原值；實際執行所有產生的 SMP，再原生選擇結果。
  間接記憶體、分支內 UBO、derivatives、discard 等不安全路徑仍拒絕。
  同時修正 SMP extended response register 解碼，涵蓋 r252 的合法四 DWORD
  回應與 r253 起的越界拒絕；沒有只截取暫存器索引低五位元。
- Compute：保留獨立 ComputeDataMaster / ComputeShader SystemC 模組，
  加入 workgroup-private modeled shared memory、resident-task barrier
  rendezvous 與真正 PCO usclib LD/ST/MUTEX 執行。API 4 接通 R32UI image2D
  load/store/atomic 與獨立 image bindings，維持 SSBO/image alias、mip/row
  padding 與 graphics → compute → graphics 的 resource generation。
- Indirect / restart：讀取真正已同步完成的 argument buffer 後進入既有
  native draw 路徑；零 count／instance 是真正 no-op。補上 bounded point
  restart，未以 host shader evaluation 或固定結果替代工作。
- Query failure：保留 terminal Failed 與 GL error，避免 Gallium 失敗後
  無限等待；不回傳成功零值，也不覆寫 caller 的 result memory。

仍未實作的 GS Transform Feedback、combined Tessellation+GS、general
divergent FS control flow、其他 image 格式／array／MS image、indirect compute、
multi-draw indirect 等，不因本輪群組清零而宣稱支援。介面與範圍詳見
[driver command contract](PVRGPU_DRIVER_COMMAND.md)。

## 原生執行與回歸證據

- Geometry 獨立稽核 7928 checks：197 Pass 中，110 案真正執行 GS，68 案為
  API/query/link/negative，12 案為真正 VS Transform Feedback，1 案為無 GS
  的 VS query，6 案為合法的不足 primitive input。110 案包括 9 案零 Emit。
  原 native-v3 的 29 Fail 全轉 Pass，168 Pass 保持。
- Tessellation 獨立稽核：400 Pass 包括 311 案真正 TES Transform Feedback、
  47 案原生 Tessellation render，以及 42 案 API/negative；不能把 400 案
  全部當成 shader execution。逐案 TCS/TES、DrawList、TF 與 pool 核對。
- Basic 41 案與 SSBO 2060 個 Pass 皆有完成的原生 compute dispatch 證據。
- 最終中央 CTest：**128/128 Pass**；Python：**83/83 Pass**。
- 專項 ASan/UBSan：GS/TEX accounting、shared/image memory、layered surface
  bounds、indirect、native TXL/texelFetch 等通過。最後的 fragment-select
  可重建測試為 compiler **49 checks**、native ISS **1436 checks**。
- Surface span 單元 **24051 checks Pass**；最新重生 texture fixtures
  與 native ISS 比對 **1490 checks Pass**。

本輪較早 checkpoint 的額外比較（不是上述最終 runtime 的完整組數字）：
indirect 20 個 live scenarios 與 llvmpipe 的 2560 DWORD／20 queries 完全相同；
graphics → image compute → graphics generation probe 兩後端皆 599 checks Pass。
各自 runtime hashes 保存在專項 receipt，未與最終 hash 混用。

## Frozen runtime 與 artifacts

| 項目 | SHA-256 |
| --- | --- |
| stock `deqp-gles31` | `9ac9ca1efae4c5194d59663e22af6c455575c1a7b795bd0024b89fc083ab128d` |
| Mesa `libgallium-26.2.1.dylib` | `bc0f893d46f7df83d54bec97a295d90aae091a2195b42cab11365be3d73bbc48` |
| `libpvrgpu_systemc_bridge.dylib` | `1e3a1106af02305ae5010317271cc3b9bf2bb6dc123fe4154a84bfa915c46566` |

原始 artifacts 根目錄：
`/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/out/runs/deqp_groups/`

- `final_api27_compute4_v1/{geometry,basic,ssbo}/`：完整 case manifest、逐案
  command/QPA/driver counters/model JSONL、summary.tsv、runtime-receipt.json。
- `final_api27_compute4_v1/geometry/independent-audit.json` 與 `audit_gs207.py`。
- `tessellation_v1/pvrgpu-final-central-v1/`：完整 406 案與 frozen receipt；
  `tessellation_v1/final-central-v1-audit.json`／`audit-final-tessellation.py`
  為獨立逐案稽核。更多說明見 [Tessellation validation](TESSELLATION_VALIDATION.md)
  與 [Geometry validation](GEOMETRY_SHADER_VALIDATION.md)。
- `final_api27_compute4_ctest.log`、`api27-python-final.log`、
  `api27-resource-final.log`、`api27-texture-fixtures-final.log`。
- `fragment_texture_select_private_v1/validation.md`、
  `texture_explicit_lod_native_v1/module-validation.md`、
  `indirect_private.a1D54q/native_v1/VALIDATION.md` 保存各專項邊界與證據。

所有執行使用 pbuffer 256×256、`rgba8888d24s8ms0`，停用 shader cache；
PvrGPU 模型使用 cache mode。未刪除失敗案例或更改比較 tolerance。
Mesa 以 pinned commit `da14d65e4499e66468094be52bff9ea0915a695e` 為基底，
PvrGPU 與歷史 llvmpipe 是不同 patch set，不能稱為相同 Mesa source tree。

## 可重現的專項入口

設定 `config/local.env` 的外部 Mesa/build 路徑後，在 repo 根目錄執行：

```sh
bash script/run_mesa_pco_compute_unit.sh all
bash script/run_mesa_pco_multisample_unit.sh
bash script/run_mesa_pco_fragment_select_unit.sh
bash script/run_mesa_resource_unit.sh all
```

這些腳本使用實際 Mesa 編譯設定，在新的外部私有目錄重建 compiler fixtures
與測試，不覆蓋正在跑 CTS 的 runtime。完整群組 runner／argv 保存於上述
artifact receipts；重跑請使用新 label，不覆寫歷史證據。

`git diff --check` 通過。另執行 source-tree cleanliness guard，仍因 repo
內既有／ignored 的 build、out、Python bytecode、Finder metadata 與
`outputs/diagnostics/rdc_debug_pixel` 而不通過，詳見
`api27-source-tree-check.log`；本輪未刪除這些檔案，也未修改使用者的
`Claude outputs/` 或測試矩陣試算表。

本輪沒有重新跑 FBO、UBO、MSAA、multisample textures、VS Transform Feedback
1320 或完整 atomic groups；先前數字仍屬各自歷史 runtime。驗證完成時變更
尚未 commit 或 GitHub push；後續發佈以 Git 紀錄為準。
