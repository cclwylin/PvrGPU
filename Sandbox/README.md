# GFXBench Manhattan Frame 0 DrawList-by-DrawList Debugger

本目錄是為 **GFXBench Manhattan 第一個 Frame（Frame 0）** 設計的「逐 DrawList 前綴除錯（Prefix Replay Debugger）」環境。

---

## 1. 核心設計原則

1. **Prefix Replay 原則 (`DrawList[0..N]`)**：
   - 不孤立執行單一 DrawList。
   - 每次測試執行從 `DrawList 0` 累積到 `DrawList N`，保留真實的 FBO、Depth Buffer、Clear 與 State 轉移。
2. **Fail-Closed 原則**：
   - 一旦在 `DrawList N` 遇到未實作的 PCO Opcode、MRT 配置或 Blend 模式，立即停止（Stop on First Error），並保存當前狀態。
3. **隔離暫存檔案**：
   - 中間生成的 PNG、FBO Dump、Json Report 與 Log 一律輸出至外部工作目錄：
     `$PVRGPU_SANDBOX_WORK_ROOT`，預設為 `$HOME/Downloads/_Codex/Working/SandBox/`
   - 不佔用 iCloud Workspace 空間。

---

## 2. Manhattan Frame 0 階段劃分

```text
[Pass 1: Shadow Depth]       -> DrawList[0 .. S] (Depth-only FBO, Viewport, Depth Write, PS=0)
[Pass 2: G-Buffer Solids]    -> DrawList[S+1 .. G] (MRT: Color0..2 + Depth, UBO, Textures)
[Pass 3: Lighting & Sky]     -> DrawList[G+1 .. L] (G-Buffer Sampling, Lighting ALU, Additive Blend)
[Pass 4: Transparents/Post]  -> DrawList[L+1 .. P] (Alpha Blending, Tone Mapping, Final Resolve)
```

---

## 3. 除錯推進順序

1. **Step 1**：建構 `DrawList[0]`（Shadow Depth Pass 的第 1 個幾何網格）。
2. **Step 2**：透過 SystemC 管線執行，輸出中間 Framebuffer/Depth PNG 至 Working 目錄。
3. **Step 3**：比對 llvmpipe Golden 結果。
4. **Step 4**：若需要修改 `src/` source code（如擴充 VDM / PCO / ISP / PBE 支持新狀態），先提出變更計畫由使用者審閱後再行修改。
