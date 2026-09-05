# PvrGPU src/ Workaround 稽核報告

日期：2026-09-03
範圍：`src/gallium/drivers/pvrgpu`(PCO Driver) 與 `src/systemc`(PVRGPU C-model)，共 99 個檔案。
方法：5 個子 agent 分頭全文讀取程式碼（而非只 grep 關鍵字），交叉核對 `PvrGPU.md`、`docs/PCO_LOWERING.md`、`docs/PVRGPU_DRIVER_COMMAND.md`、driver 自己的 `README.md`。

## 一句話結論

Driver 端（`pvrgpu_context.c` / `pvrgpu_pco.c`）大量存在「用 dEQP/benchmark 測試名稱或 shader hash 當作查表 key，直接回傳預先算好的假數據」的做法，而且**driver 自己的 README 明文禁止這種做法**，屬於真正的 workaround，不是合理簡化。C-model（`systemc/`）本身的品質相對高，多數模組是誠實的 fail-closed 或誠實的空 stub；但也有一個嚴重例外（`pco_iss.cpp` 的 conditionals shader 手刻捷徑）。而你稍早遇到的 `line_loop_capture` FAIL，根因已經找到，見下方「FAIL 根因」。

---

## 🔴 FAIL 根因：`0001_primitives_line_loop_capture` 的 6 個 unsupported draw event

這不是單一 bug，而是兩層問題疊加：

**第一層（表面）：這條路徑之前從沒有真的被測試過。**
現有的「支援」其實是假的——`pvrgpu_context.c` 裡有一整批以 `PVRGPU_RDC_CASE_NAME`（讀自環境變數，即目前跑的 dEQP case 名稱）做 `strcmp` 比對的查表：
- `pvrgpu_deqp_rasterization_primitives_profile()`（`pvrgpu_context.c:815-949`）針對 `"line_loop"`、`"line_loop_wide"`、`"line_strip"`、`"lines"`、`"points"` 等**精確字串**回傳寫死的 `ia_vertices`/`ia_primitives`/`vs_invocations`/... 等 counter 數值（例如第 819 行 `{"line_loop", 3, 3, 4, MESA_PRIM_LINE_LOOP, true, 12, 12, 12, 12, 12, 0, 1052}`）。
- 你這次跑的 capture 名稱是 `"0001_primitives_line_loop_capture"`，和表裡任何字串都對不上（已用 grep 全域確認零命中），所以查表 fallback 失效，才第一次真正落到「沒有實作」的路徑，被 `pvrgpu_note_unsupported_draw()`（`:13076-13133`）誠實地記成 unsupported。

**第二層（真因）：就算查表命中、繞過表面問題，LINE_LOOP 在整條 pipeline 裡本來就無法正確工作。**
- `vertex_fetch.cpp:240-269` 的 `ExpandTopology()` 把每條 line / point 轉成「面積必為 0」的退化三角形（line: `a,b,b`；line_loop 收尾用 `last,first,first`），然後把 `state.draw.topology` 直接改寫成 `kTriangleList`（`:501`），下游（ClipCull、ParameterBuffer、Tiler、ISP、PBE）從此完全不知道這原本是線段。
- `clip_cull.cpp:764-766` 對零面積三角形呼叫 `reject_degenerate=true` 的 `ClassifyFrontFacing`，直接丟出 `"ClipCull received a degenerate input triangle"`——錯誤訊息會讓人誤以為是三角形資料壞掉，而不是「線段/點根本沒實作」。
- 另一條路徑（需要 near-plane clip 時）則是 `clip_cull.cpp:424` + `tiler.cpp:78-79` 把這種三角形標成 `rasterizable = 0` 靜默跳過——畫面上什麼都沒畫出來，但 draw call「看起來」成功了。
- 全 `systemc/` 搜尋確認：沒有任何 line-width 或 point-size 的展開邏輯。

**結論與建議：** 不要再幫 `0001_primitives_line_loop_capture` 加一筆查表項目去「修好」它——那只是繼續踩 README 明文禁止的地雷。真正該做的是在 `vertex_fetch.cpp` / `clip_cull.cpp` 實作線段與點的正確幾何展開（寬度/大小、不退化成零面積三角形），並讓 `pvrgpu_array_primitive_triangle_count()`（`pvrgpu_context.c:11622-11640`，目前只認 TRIANGLES/STRIP/FAN）能真正處理 LINE_LOOP。這也代表：**現有所有「line_loop / line_strip / lines / points 已通過」的 dEQP 結果都不可信**，因為它們是查表出來的假結果，不是模型真的算出來的。

---

## 依嚴重度排列的 Workaround 清單

### 🔴 高風險（會讓人誤以為功能正確，實際是假的/錯的）

1. **`pvrgpu_context.c`：整套「依測試名稱查表回傳假 counter/像素結果」子系統**
   分布在約 425–2400 行（十幾張查表，`strcmp(PVRGPU_RDC_CASE_NAME, ...)`），呼叫點延伸到 ~14300 行。除了上面提到的 primitives 查表，還包含：
   - 貼圖 filtering 的每格式/每組合硬編 texel-fetch 次數（`:623-789`，如 `texels_2d_bgra8[]`、`combination_profiles[]`）。
   - 針對 `"terrain.terrain.capture.1"` 場景整段手工重放邏輯，含依 framebuffer 寬度硬編數字（`requested_width == 800u ? 51u : 42u`，`:13264-13269`）。
   - 針對 `"shadow.shadow.capture.1"` 的類似機制（`:13323+`）。
   - `pvrgpu_negative_coverage_transparent_framebuffer_case()`（`:475-482`）：針對特定 dEQP negative-coverage 測試強制把 alpha 設成 0。
   - `pvrgpu_case_suppresses_draw_commands()`（`:465-473`）：對單一 dEQP 測試名稱做 `strcmp`，直接吃掉所有 draw command。

   **這為什麼是真正的 workaround，而不是合理簡化：** driver 自己的 `README.md:45` 白紙黑字寫「Do not extend this driver by special-casing captured test names. The driver boundary must be normal Gallium state.」——這整套機制正是這句話明令禁止的事。目前「通過」的 dEQP/benchmark 結果，有相當比例是被這套查表餵出來的罐頭答案，不代表底層模型真的算對。

2. **`pvrgpu_deqp_tessellation_profiles.h` + `pvrgpu_context.c:1988-2024`：pipeline-statistics counter 造假**
   整支 header 是用**dEQP case 名稱字串**當 key 的靜態表（如 `"dEQP-GLES31.functional.tessellation.common_edge.quads_equal_spacing"`），存著 `hs_invocations`/`ds_invocations`/`vs_invocations` 等預先算好的數字。`strcmp` 命中就直接回傳表裡的數字，完全不管 SystemC 模型這次跑出來的 tessellation invocation 數到底對不對。這是「pattern-match 已知測試、回傳罐頭答案」的教科書案例，而且沒有任何註解說明這是假數據。

3. **`pvrgpu_pco.c` / `pvrgpu_pco.h`：PCO 編譯後端只認識約 20 支寫死的已知 benchmark shader**
   `pvrgpu_pco_compile_{lit_mesh,texture,refract,shadow,terrain,ideas,conditionals,color_triangle}()` 等每一個對外 API，都先比對 shader 原始碼的 BLAKE3 hash（`pvrgpu_source_hash_matches()`，`:792-807`）再比對 NIR 的精確 instruction/op 數量（例如 `pvrgpu_validate_texture_nir()` 要求 `instructions == 41`、`fmul == 7`…否則直接 fail-closed）。命中之後確實有跑真正的 PCO pipeline（不是回傳固定 binary），所以不是 100% 造假，但「texture profile」、「terrain profile」這種命名和文件敘述，讀起來像是支援一整類 shader，實際上一類只認一支位元組完全相同的 shader，任何非該 fixture 的 shader 一律 reject。這點在 `pvrgpu_pco.h` 有註明（`:141,176,259-260` 都寫了 fail-closed），屬於「有揭露但範圍名不符實」。

4. **`systemc/shader/pco_iss.cpp`：conditionals shader 的手刻捷徑，直接繞過真正的 decoder**
   `DecodePcoProgram()`（~line 5061）對整支 binary 做一次 `==` 比對（`kConditionalsVertexBinary`/`kConditionalsFragmentBinary`，`:446+`），命中就呼叫 `BuildConditionalsInstructions()`（`:2604`）——**完全跳過共用的 `DecodeHeader()`/decoder**，改用工程師自己手寫、對這支 binary 位元組意義的「理解」去手動組出 ~48/43 筆 `PcoInstruction`（含寫死的暫存器編號與立即數如 `0x40400000`）。這組手刻語意從未和真正的 bit-field decode 交叉驗證過，一旦這個人工理解有誤，會在完全沒有任何錯誤訊號的情況下算錯這支 shader。更嚴重的是：檔案開頭註解明文寫著「no project-local shader-name shortcut exists」，而這正是一個 shader-name/binary shortcut。Fragment 執行路徑另外還有一個獨立的、與通用執行器並行、彼此從未互相驗證的 mini-interpreter（`:6220-6293`），重複實作了 FADD/FMUL/FMAD/FGE/CSEL 等語意。

### 🟡 中風險（有缺口，但揭露方式老實 / 影響範圍有限）

5. `pvrgpu_screen.c:175`：`draw_indirect` cap 無條件回報 `true`，但 `pvrgpu_context.c` 裡真正能處理 indirect draw 的路徑只存在於測試名稱查表命中的少數 benchmark slice，一般 `glDrawArraysIndirect` 極可能直接落入 unsupported。建議降低此 cap 或補上通用實作。
6. `pvrgpu_pco.c:3818-3843`：terrain D1 highp lowering 雖然註解特別強調「不是 shader-name switch」，但仍然掛在同一組 source-hash 表下，只認一支固定 shader；註解和實際範圍有落差，但邏輯本身是真的結構化 lowering，不是罐頭輸出。
7. `pco_iss.cpp`：`kBranch`/`kBranchConditional`/`kLoopBegin`/`kLoopEnd` 有完整的 dispatch 邏輯，但目前沒有任何 decoder 路徑會真的產生這些 opcode——屬於未驗證的死碼，非立即風險。
8. `systemc/texture/texture_unit.cpp:170`：`kGlbenchTrilinear04CoordinateRoundThreshold = 0.5F - 1/256` 只在 `functional_case == kFillTexTrilinearLinear04` 時套用，沒有推導依據的註解，形狀上像是為了配合單一 benchmark case 微調出來的數字；但 `PvrGPU.md` 本身也把這個 case 標成 `PENDING`（尚未宣稱驗證通過），所以不是「假裝已驗證」，只是提醒之後驗證時要特別注意這個常數。

### 🟢 誠實揭露、不算 workaround（給你對照，避免誤判）

- `pvrgpu_screen.c` 的 `pvrgpu_fence_finish()` 永遠回傳完成——但 README 有寫明這是因為 driver 目前同步 flush，是揭露過的簡化。
- `pvrgpu_resource.c` 的 `pvrgpu_emit_unsupported_resource_op/copy_region/blit`：老實地計數並回報 unsupported，不是造假。
- `systemc/` 裡的多數小檔案（`data_master/*`、`host/*`、`firmware/*`、`compression/*`、`cache_mmu/mmu_bif.*`、`memory/{mem_fabric,on_chip_fabric}.*`）是空 stub，但每個都有註解說明「目前無真實行為」，不是假裝有實作。
- `cache_mmu`（cache_array / mixed_cache / usc_l2_cache / texture_cache / slc）、`memory/{dram_address_space,dram_model,gpu_memory_system}`、`common/functional_types`、`texture_unit.cpp` 其餘部分：讀完整份程式碼後認定是紮實、fail-closed 的真實功能模型，DRAM 固定 1-cycle 延遲等簡化也在 `PvrGPU.md:153` 有明文揭露為「使用者選擇的 functional/performance model 假設」。
- `pco_decoder.cpp` / `usc_cluster.cpp` / `usc_slot.cpp`：所有 unsupported case 都丟 `std::runtime_error`，沒有發現捷徑。
- PCO 編譯後端與 lowering test 沒有任何 TODO/FIXME/HACK/XXX 或 `assert(0)`，測試本身用 golden hash 誠實鎖住目前的（狹窄）行為。

---

## 建議優先順序

1. **先處理 FAIL**：在 `vertex_fetch.cpp`/`clip_cull.cpp` 加入真正的 line/point 幾何展開，讓 `pvrgpu_array_primitive_triangle_count()` 認得 LINE_LOOP 等 topology，而不是繼續往查表加條目。
2. **盤點並標記 `pvrgpu_context.c` 裡所有 `PVRGPU_RDC_CASE_NAME` 查表**：這是目前唯一違反專案自己 README 規則的地方，建議列一份清單，逐一評估「拔掉查表後這個 dEQP case 是否還真的過」，凡是過不了的，都代表之前的「PASS」是假的。
3. **`pco_iss.cpp` 的 conditionals 手刻捷徑**：這是唯一在「數值正確性」上有實質風險且沒有被誠實揭露的 C-model 問題，建議優先讓它改走共用 decoder，或至少加上與真實 decode 結果的交叉驗證。
4. `pvrgpu_pco.c` 的 hash-locked profile 機制、`draw_indirect` cap 誠實性，屬於範圍名實不符，優先度較低但建議在文件/命名上澄清，避免誤導後來的維護者。

---

*本報告由 5 個獨立 agent 分別全文讀取 `src/gallium/drivers/pvrgpu`（PCO Driver）與 `src/systemc`（C-model）共 99 個檔案後彙整而成，所有引用的行號均來自各 agent 實際讀取程式碼後回報，非臆測。*
