// VertexPdsEngine：PDS（Programmable Data Sequencer）的頂點著色器分支。
#include "pds/vertex_pds_engine.h"

#include "common/functional_types.h"

namespace pvrgpu::stub {

VertexPdsEngine::VertexPdsEngine(sc_core::sc_module_name name, MemoryPool &pool)
    : sc_module(name), pool_(pool) {
  SC_THREAD(Run);
}

void VertexPdsEngine::Run() {
  while (true) {
    const PipelineTxn txn = input.read();
    PipelineState state = LoadPipelineState(pool_, txn.state);
    RequireStage(state.stage, PipelineStage::kVertexFetched, name());

    state.stage = PipelineStage::kVertexPdsReady;
    StorePipelineState(pool_, txn.state, state);
    output.write(txn);
  }
}

}  // namespace pvrgpu::stub
