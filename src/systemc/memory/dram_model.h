// DramModel：DRAM = Dynamic Random-Access Memory（動態隨機存取記憶體）。
// 它是 SLC（System Level Cache，系統層級快取）下方的同一個真實
// backing model：UploadRun 在 shader sample 前預置完整 texture allocation；
// TextureRun 回應 TCU/SLC miss；Run 處理 framebuffer writeback 與獨立
// PNG readback。Backing 以 sparse 128-byte pages 保存不連續 texture/framebuffer
// GPU addresses，不會為中間空洞配置巨大 vector。每 request 使用固定一個
// cycle 的 event-driven latency；FIFO 僅傳 control/MemoryPool handle，最終
// RGBA 一定由新的 DRAM readback handle 交給 JsonReporter。
#pragma once

#include "common/pipeline_state.h"

#include <systemc>

#include <cstddef>
#include <cstdint>
#include <array>
#include <map>
#include <vector>

namespace pvrgpu::stub {

inline constexpr std::uint64_t kDramFixedLatencyCycles = 1;

class DramModel final : public sc_core::sc_module {
public:
  sc_core::sc_fifo_in<MemoryTxn> input{"input"};
  sc_core::sc_fifo_out<PipelineTxn> output{"output"};
  sc_core::sc_port<sc_core::sc_fifo_in_if<MemoryTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      texture_input{"texture_input"};
  sc_core::sc_port<sc_core::sc_fifo_out_if<MemoryTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      texture_output{"texture_output"};
  sc_core::sc_port<sc_core::sc_fifo_in_if<MemoryTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      upload_input{"upload_input"};
  sc_core::sc_port<sc_core::sc_fifo_out_if<MemoryTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      upload_output{"upload_output"};

  DramModel(sc_core::sc_module_name name, MemoryPool &pool);

private:
  void Run();
  void TextureRun();
  void UploadRun();
  void EnsureBackingRange(std::uint64_t address, std::size_t bytes);
  void WriteBacking(std::uint64_t address,
                    const std::uint8_t *source,
                    std::size_t bytes);
  std::vector<std::uint8_t> ReadBacking(std::uint64_t address,
                                        std::size_t bytes) const;

  MemoryPool &pool_;
  static constexpr std::size_t kBackingPageBytes = 128;
  std::map<std::uint64_t, std::array<std::uint8_t, kBackingPageBytes>>
      backing_pages_;
};

} // namespace pvrgpu::stub
