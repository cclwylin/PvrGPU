# Geometry Shader 原生實作與 dEQP 驗證

2026-09-08，`native-v3`。這是已可執行的第一階段 GS 實作，**尚未 all pass，也尚未追平 llvmpipe**。

## 完整 207-case 結果

範圍是 stock CTS 的 `dEQP-GLES31.functional.geometry_shading.*`，不刪除失敗或不支援案例。

| 執行版本 | Pass | Fail | NotSupported |
| --- | ---: | ---: | ---: |
| PvrGPU 原始基線 | 99 | 98 | 10 |
| PvrGPU native-v1 | 158 | 39 | 10 |
| PvrGPU native-v2 | 167 | 30 | 10 |
| PvrGPU native-v3 | 168 | 29 | 10 |
| llvmpipe 基線 | 206 | 0 | 1 |

逐例比較原始基線：69 個 Fail 轉為 Pass，原有 99 個 Pass 全部保持，沒有狀態退步。

不能把 168 個 QPA Pass 全部當成原生 GS 測試：

- 83 個 Pass 有完成的 SystemC GS invocation 證據，包含 9 個真正執行但零 Emit 的案例；共有 688 次 GS invocation、6,069 個 emitted vertices、4,423 個完整 output primitives。
- 68 個是 query/API/link 或 negative API 案例。
- 15 個是合理的無 GS 執行案例：6 個 adjacency 輸入不足、1 個無 GS query、8 個 VS transform-feedback。
- 2 個 VS indirect transform-feedback 案例仍在 draw 被拒絕後取得 QPA Pass，**不計為有效原生 indirect 支援證據**。

5 個 primitives-generated query 的 QPA、driver 回傳值與完成的模型 counter 均相同：無 GS 8、無 amplification 8、amplification 24、partial primitives 8、instanced 32。零 Emit 不以 input primitive 數代替 output 數。

## 架構與本輪修正

`GeometryShader` 有獨立 `.h/.cpp`、event-driven `SC_THREAD`、bounded FIFO 與 pool-backed payload，位於 VS 與 ClipCull 之間。GS 不是 VS/CS 別名；Compute 與 Tessellation 模組仍獨立。後續 API25 的原生 Tessellation 接線與驗證見 [Tessellation validation](TESSELLATION_VALIDATION.md)。

Mesa 保留 `MESA_SHADER_GEOMETRY`，產生原生 PCO。模型執行真正的 LD/WDF、UVSW WRITE/EMIT/CUT/ENDTASK，primitive input staging/read 經 modeled GPU memory；沒有 host NIR 解譯、預先計算的顏色或依測試名稱選擇結果。

Input adjacency、strip parity、last provoking vertex、Emit/EndPrimitive 與不足 primitive 的處理參考 pinned Mesa/llvmpipe；flat varying 保留 raw DWORD。另補上三階段 UBO transport、動態 CB0、PrimitiveID/InvocationID、零屬性 VS、零 Emit，以及空白 FS 的原生 `NOP.end`。API24 深拷貝 shader/resource payload，舊版本在讀取新增欄位前拒絕。

空白 FS 不合成顏色 export，NOP 記為 control、非 ALU。API 測試以不同 clear color 加非均勻 initial attachment bytes，在 cache 模式逐位元組驗證 LOAD 保持，並要求真 FS invocation、零 color read/blend/write。

詳細欄位與支援邊界見 [driver command contract](PVRGPU_DRIVER_COMMAND.md)。

## 尚未通過的範圍

| Fail 分類 | 案例數 | 已確認的缺口 |
| --- | ---: | --- |
| Layered rendering / instanced layered | 24 | layered attachment LOAD/layout 尚未支援，後續驗證 draw 的 array/3D/cube sampling 亦有缺口 |
| GS texture resources | 2 | GS sampling 尚未接通 |
| Primitive restart | 1 | 純 assembly helper 已測試，但 live driver restart transport 尚未接通 |
| VS transform-feedback overflow readback | 2 | 既有非 GS 路徑的 readback 驗證失敗 |

10 個 NotSupported：1 個要求 2,048 output components（兩後端共同不支援）、1 個要求 point size 5、8 個要求 multisample 2D-array extension。後兩類共 9 個是 PvrGPU 與 llvmpipe 的額外差距。

本輪沒有觀察到已完成原生 GS 執行的案例發生 raster mismatch；這不代表未接通的功能已正確。所有 GS MRT 目前明確拒絕：API 尚未攜帶每個 attachment 的獨立初始 LOAD，即使 FS 宣告全部 targets，未覆蓋像素也不能從 clear color 重建。

## 驗證與重現證據

- CTest：**82/82 Pass**，`ctest-v4.log`。
- Python：**80/80 Pass**，`python-regression-v4.log`。
- Native GS ISA/module：**18,042 checks Pass**，亦有 ASan/UBSan receipt。
- GS API24：**1,882 checks Pass**；6 組真實 compiler pipeline、snapshot ownership、version guard、query generation、zeroVBO、zeroEmit、emptyFS 與 MRT 負例。
- Graphics statistics：**13 checks Pass**。
- GS compiler：中央 **153 checks Pass**；含 fixture 檔案 I/O 的完整 compiler 單元為 **197 checks Pass**，並通過既有 VS/FS/CS compiler regression。
- Primitive assembly 與實際 Mesa helper 的 differential + ASan/UBSan：**4,562,202 checks Pass**。

空白 CS 與 FS 的獨立編譯結果恰好都是相同的 8-byte `NOP.end`；測試確認逐 byte 相同、FS ownership 與零副作用，不再錯誤要求這個共用 ISA 指令一律不能作 FS 解碼。非空 Compute/atomic 程式的跨階段拒絕測試仍保留。

Stock CTS SHA256：`9ac9ca1efae4c5194d59663e22af6c455575c1a7b795bd0024b89fc083ab128d`。
207-case manifest SHA256：`11a624dccc5d33c106d4868637c5fa858e3040993d398306137fc2f3deed7d2a`。
兩個 Mesa source 皆以 `da14d65e4499e66468094be52bff9ea0915a695e` 為基底，但**不是相同 source tree**：llvmpipe 原本已套 zero-output statistics patch；本輪 native GS backend patch 只套於 PvrGPU。

獨立稽核重新核對 414 份 QPA 的 CasePath/result、相同 CTS hash/flags、renderer/DYLD 載入路徑與兩後端 runtime 前後 hash。PvrGPU 的 163 筆完成記錄、86 次 physical submissions 均核對；pool 共 4,916 allocation / 4,916 release，leaks=0。PvrGPU 與 llvmpipe 的 QPA 狀態相同 169/207；這是測試狀態比較，不是額外宣稱兩後端所有 framebuffer bytes 完全相同。

完整 artifacts 位於工作區外的：

`/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/out/runs/deqp_groups/geometry_shading_v1/`

其中 `run_baseline.py` 保存 stock CTS runner/環境與逐例隔離方式，`pvrgpu-native-v3/` 保存本輪逐例原始結果與 frozen receipt，`native-v3-independent-audit.json` / `native-v3-independent-verification.md` 保存獨立分類；compiler、assembly、native module 的專項報告亦在此目錄。

本輪未重新執行所有先前 FBO、MSAA、atomic、SSBO 或 Compute 的完整 dEQP groups；82 個回歸測試不能替代那些完整 CTS 結果。
