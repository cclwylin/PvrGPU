// TPU texture-response demultiplexer. All shader stages may issue onto the
// ordered TPU -> TCU request FIFO, while each stage owns a distinct return
// FIFO so one worker can never consume another worker's response.
#pragma once

#include "model_types.h"

#include <systemc>

#include <stdexcept>

namespace pvrgpu::stub {

class TextureResponseRouter final : public sc_core::sc_module {
 public:
  sc_core::sc_fifo_in<MemoryTxn> input{"input"};
  sc_core::sc_port<sc_core::sc_fifo_out_if<MemoryTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      vertex_output{"vertex_output"};
  sc_core::sc_port<sc_core::sc_fifo_out_if<MemoryTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      fragment_output{"fragment_output"};
  sc_core::sc_port<sc_core::sc_fifo_out_if<MemoryTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      compute_output{"compute_output"};
  sc_core::sc_port<sc_core::sc_fifo_out_if<MemoryTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      geometry_output{"geometry_output"};
  sc_core::sc_port<sc_core::sc_fifo_out_if<MemoryTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      tessellation_control_output{"tessellation_control_output"};
  sc_core::sc_port<sc_core::sc_fifo_out_if<MemoryTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      tessellation_evaluation_output{"tessellation_evaluation_output"};

  explicit TextureResponseRouter(sc_core::sc_module_name name)
      : sc_core::sc_module(name) {
    SC_THREAD(Run);
  }

 private:
  using OutputPort =
      sc_core::sc_port<sc_core::sc_fifo_out_if<MemoryTxn>, 0,
                       sc_core::SC_ZERO_OR_MORE_BOUND>;

  void Write(OutputPort &output, const MemoryTxn &response) {
    if (output.size() == 0)
      throw std::runtime_error(
          "TextureResponseRouter response route is not connected");
    output->write(response);
  }

  void Run() {
    while (true) {
      const MemoryTxn response = input.read();
      switch (response.response_route) {
      case MemoryResponseRoute::kTextureVertex:
        Write(vertex_output, response);
        break;
      case MemoryResponseRoute::kTextureFragment:
        Write(fragment_output, response);
        break;
      case MemoryResponseRoute::kTextureCompute:
        Write(compute_output, response);
        break;
      case MemoryResponseRoute::kTextureGeometry:
        Write(geometry_output, response);
        break;
      case MemoryResponseRoute::kTextureTessellationControl:
        Write(tessellation_control_output, response);
        break;
      case MemoryResponseRoute::kTextureTessellationEvaluation:
        Write(tessellation_evaluation_output, response);
        break;
      case MemoryResponseRoute::kNone:
        throw std::runtime_error(
            "TextureResponseRouter received an unrouted response");
      default:
        throw std::runtime_error(
            "TextureResponseRouter received an invalid response route");
      }
    }
  }
};

}  // namespace pvrgpu::stub
