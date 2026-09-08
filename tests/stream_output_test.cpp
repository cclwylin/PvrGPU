// Native VS exports -> independent depth-one StreamOutput -> modeled memory.
#include "common/stream_output_types.h"
#include "geometry/stream_output.h"
#include "memory/gpu_memory_system.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

using namespace pvrgpu::stub;
namespace {
unsigned checks = 0;
void Check(bool good, const char *message) {
  ++checks;
  if (!good) throw std::runtime_error(message);
}
template <typename F> void Reject(F call) {
  try { call(); } catch (const std::runtime_error &error) {
    Check(std::string(error.what()).find("stream output") != std::string::npos,
          "rejection comes from stream-output contract"); return;
  }
  throw std::runtime_error("invalid stream-output contract was accepted");
}
StreamOutputTarget Target(unsigned buffer, unsigned resource, unsigned offset,
                          unsigned size, unsigned cursor, unsigned stride) {
  StreamOutputTarget target;
  target.output_buffer = buffer; target.resource_token = resource;
  target.target_token = 100 + buffer;
  target.gpu_address = UINT64_C(0x500000000) + resource * 0x10000;
  target.bytes_size = 512; target.buffer_offset = offset;
  target.buffer_size = size; target.internal_offset = cursor;
  target.stride_dwords = stride;
  return target;
}
void Validation() {
  const std::vector<StreamOutputBinding> bindings{{0, 4, 0, 0, 0}};
  const std::vector<StreamOutputTarget> targets{Target(0, 1, 0, 512, 0, 4)};
  ValidateStreamOutputLayout(bindings, targets, 4);
  ValidateStreamOutputLayout(bindings, {}, 4); // unbound target is overflow
  for (unsigned mutation = 0; mutation < 22; ++mutation) {
    auto bad = targets;
    switch (mutation) {
    case 0: bad[0].output_buffer = 4; break;
    case 1: bad.push_back(bad[0]); break;
    case 2: bad[0].resource_token = 0; break;
    case 3: bad[0].target_token = 0; break;
    case 4: bad[0].gpu_address = 0; break;
    case 5: ++bad[0].gpu_address; break;
    case 6: bad[0].bytes_size = 0; break;
    case 7: bad[0].bytes_size = kMaximumStreamOutputResourceBytes + 1; break;
    case 8: bad[0].gpu_address = UINT64_MAX - 3; break;
    case 9: bad[0].buffer_offset = 1; break;
    case 10: bad[0].internal_offset = 1; break;
    case 11: bad[0].buffer_offset = 516; break;
    case 12: bad[0].buffer_size = 513; break;
    case 13: bad[0].internal_offset = 516; break;
    case 14: bad[0].readback = {0, 1}; break;
    case 15: bad[0].stride_dwords = 3; break;
    case 16: bad.push_back(Target(1, 1, 0, 512, 0, 4)); ++bad[1].gpu_address; break;
    case 17: bad.push_back(Target(1, 1, 0, 512, 0, 4)); bad[1].target_token = bad[0].target_token; break;
    case 18: bad[0].stride_dwords = 0; break;
    case 19: bad[0].stride_dwords = 65; break;
    case 20: bad.push_back(Target(1, 2, 0, 512, 0, 4)); bad[1].gpu_address = bad[0].gpu_address; break;
    case 21: bad.push_back(Target(1, 2, 0, 512, 0, 4)); bad[1].gpu_address = bad[0].gpu_address + 508; break;
    }
    Reject([&] { ValidateStreamOutputLayout(bindings, bad, 4); });
  }
  for (unsigned mutation = 0; mutation < 11; ++mutation) {
    auto bad = bindings;
    switch (mutation) {
    case 0: bad[0].stream = 1; break;
    case 1: bad[0].output_buffer = 4; break;
    case 2: bad[0].num_components = 0; break;
    case 3: bad[0].num_components = 5; break;
    case 4: bad[0].output_dword = 1; break;
    case 5: bad[0].output_dword = UINT32_MAX; break;
    case 6: bad[0].dst_offset_dwords = UINT32_MAX; break;
    case 7: bad.resize(65, bad[0]); break;
    case 8: bad.push_back({0,1,0,3,0}); break;
    case 9: bad[0].output_buffer = 3; bad[0].dst_offset_dwords = 61; break;
    case 10: bad.clear(); break;
    }
    Reject([&] { ValidateStreamOutputLayout(bad, targets, 4); });
  }
  Reject([&] { ValidateStreamOutputLayout(bindings, targets, 65); });
  std::vector<StreamOutputBinding> maximum;
  std::vector<StreamOutputTarget> four;
  for (unsigned buffer = 0; buffer < 4; ++buffer) {
    four.push_back(Target(buffer, 1 + buffer, 0, 512, 0, 16));
    for (unsigned c = 0; c < 16; ++c) maximum.push_back({buffer * 16 + c, 1, buffer, c, 0});
  }
  ValidateStreamOutputLayout(maximum, four, 64);
  auto huge = targets;
  huge[0].bytes_size = huge[0].buffer_size = kMaximumStreamOutputResourceBytes;
  huge[0].internal_offset = huge[0].buffer_size;
  ValidateStreamOutputLayout(bindings, huge, 4); // legal full append target
  auto missing = bindings; missing[0].output_buffer = 3;
  ValidateStreamOutputLayout(missing, targets, 4);
  auto adjacent = targets;
  adjacent.push_back(Target(1, 2, 0, 512, 0, 4));
  adjacent[1].gpu_address = adjacent[0].gpu_address + adjacent[0].bytes_size;
  ValidateStreamOutputLayout(bindings, adjacent, 4);
}

struct TestDraw {
  PrimitiveTopology topology;
  std::vector<std::vector<unsigned>> primitives;
  std::vector<StreamOutputBinding> bindings;
  std::vector<StreamOutputTarget> targets;
  unsigned admitted;
};
std::vector<VertexLane> NativeLanes() {
  const auto program = DecodePcoProgram(ShaderStage::kVertex, AttributeFetchVertexPcoBinary());
  std::vector<VertexLane> lanes(7);
  const std::uint32_t special[] = {0x7fc12345, 0x80000000, 0xff800000, 0x00000001,
                                  0x3f800000, 0x7f800000, 0xffffffff};
  for (unsigned i = 0; i < lanes.size(); ++i) {
    const auto execution = ExecuteVertexPco(program.summary, program.instructions,
        std::vector<std::uint32_t>{special[i], 0x3e800000U + i});
    Check(execution.emitted && execution.ended_task && execution.written_mask == 15,
          "source is genuinely executed native VS VTXOUT");
    std::copy(execution.outputs.begin(), execution.outputs.end(), lanes[i].vertex_output);
    lanes[i].emitted = execution.emitted; lanes[i].ended = execution.ended_task;
  }
  return lanes;
}

struct Pending {
  PipelineTxn txn;
  std::map<std::uint64_t, std::vector<std::uint8_t>> expected;
  std::vector<StreamOutputTarget> targets;
  unsigned written, needed;
};
Pending Make(MemoryPool &pool, GpuMemorySystem &memory, const TestDraw &draw,
             unsigned sequence) {
  Pending pending;
  pending.written = draw.admitted; pending.needed = static_cast<unsigned>(draw.primitives.size());
  pending.targets = draw.targets;
  const auto lanes = NativeLanes();
  for (const auto &target : draw.targets) {
    auto inserted = pending.expected.emplace(target.resource_token,
                                             std::vector<std::uint8_t>(target.bytes_size, 0xa5));
    if (inserted.second) memory.HostWrite(target.gpu_address, inserted.first->second.data(), target.bytes_size);
  }
  std::vector<VertexLaneRef> refs;
  unsigned index = 0;
  for (const auto &primitive : draw.primitives) {
    Check(!primitive.empty() && primitive.size() <= 3, "test has real primitive connectivity");
    for (unsigned v = 0; v < 3; ++v) refs.push_back({primitive[std::min<std::size_t>(v, primitive.size() - 1)], index++});
  }
  // Independent oracle: admitted primitive count is explicit in each case,
  // then raw bytes are scattered to the declared target offsets.
  for (unsigned p = 0; p < draw.admitted; ++p) {
    for (unsigned vertex : draw.primitives[p]) {
      for (auto &target : pending.targets) {
        bool used = false;
        for (const auto &binding : draw.bindings) {
          if (binding.output_buffer != target.output_buffer) continue;
          used = true;
          auto &bytes = pending.expected.at(target.resource_token);
          const auto offset = target.buffer_offset + target.internal_offset + binding.dst_offset_dwords * 4;
          std::memcpy(bytes.data() + offset, lanes[vertex].vertex_output + binding.output_dword,
                      binding.num_components * 4);
        }
        if (used) target.internal_offset += target.stride_dwords * 4;
      }
    }
  }
  PipelineState state;
  state.stage = PipelineStage::kVertexShaded; state.memory_mode = memory.mode();
  state.draw.topology = PrimitiveTopology::kTriangleList; state.source_topology = draw.topology;
  state.draw.vertex_count = static_cast<std::uint32_t>(refs.size());
  state.vertex_pco_abi.vertex_outputs = 4;
  // Deliberately no static written mask: undefined exports cannot turn a
  // structurally legal transform-feedback binding into an API rejection.
  state.vertex_lanes = StoreNewArray(pool, lanes); state.vertex_lane_refs = StoreNewArray(pool, refs);
  state.stream_output_bindings = StoreNewArray(pool, draw.bindings);
  state.stream_output_targets = StoreNewArray(pool, draw.targets);
  pending.txn.sequence = sequence; pending.txn.state = pool.Allocate(sizeof(state));
  StorePipelineState(pool, pending.txn.state, state);
  return pending;
}
void Verify(MemoryPool &pool, GpuMemorySystem &memory, const Pending &pending) {
  const auto state = LoadPipelineState(pool, pending.txn.state);
  Check(state.stage == PipelineStage::kVertexShaded && state.stream_output_complete,
        "SO completes without changing the downstream raster stage");
  Check(state.stream_output_primitives_written == pending.written &&
        state.stream_output_primitives_storage_needed == pending.needed,
        "actual whole-primitive written and storage-needed counts");
  const auto targets = LoadArray<StreamOutputTarget>(pool, state.stream_output_targets);
  for (unsigned i = 0; i < targets.size(); ++i) {
    Check(targets[i].internal_offset == pending.targets[i].internal_offset,
          "independent target append cursor matches admitted vertices only");
    const auto actual = LoadArray<std::uint8_t>(pool, targets[i].readback);
    const auto &expected = pending.expected.at(targets[i].resource_token);
    Check(actual.size() == expected.size(), "whole resource readback extent");
    for (unsigned byte = 0; byte < expected.size(); ++byte)
      Check(actual[byte] == expected[byte], "raw DWORD payload, holes, guards and alias writes preserved");
    Check(memory.backing().Read(targets[i].gpu_address, actual.size()) == actual,
          "snapshot comes from flushed modeled backing, not host reconstruction");
  }
  if (!targets.empty()) {
    if (memory.mode() == MemoryMode::kDirect)
      Check(state.counters.memory_direct_read_bytes != 0 && !state.counters.dram_cycles,
            "direct readback has actual traffic without DRAM latency");
    else Check(state.counters.dram_read_bytes != 0 && state.counters.dram_cycles != 0,
               "modeled readback contributes actual DRAM traffic and latency");
  }
  ReleaseFunctionalPayloads(pool, state); pool.Release(pending.txn.state);
}
}  // namespace

int sc_main(int argc, char **argv) {
  try {
    const std::string mode = argc > 1 ? argv[1] : "direct";
    Check(mode == "direct" || mode == "bypass" || mode == "cache", "recognized memory mode");
    const auto memory_mode = mode == "direct" ? MemoryMode::kDirect :
        mode == "bypass" ? MemoryMode::kBypass : MemoryMode::kCache;
    Validation();
    MemoryPool pool; GpuMemorySystem memory(memory_mode);
    StreamOutput module("stream_output", pool, &memory);
    sc_core::sc_fifo<PipelineTxn> input("input", 1), output("output", 1);
    module.input(input); module.output(output);
    const std::vector<StreamOutputBinding> full{{0, 4, 0, 0, 0}};
    std::vector<TestDraw> cases{
      {PrimitiveTopology::kPoints, {{0},{1},{2}}, full, {Target(0,1,16,256,0,4)}, 3},
      {PrimitiveTopology::kLines, {{0,1},{2,3}}, full, {Target(0,1,16,256,16,4)}, 2},
      {PrimitiveTopology::kLineStrip, {{0,1},{1,2},{2,3}}, full, {Target(0,1,16,256,0,4)}, 3},
      {PrimitiveTopology::kLineLoop, {{0,1},{1,2},{2,0}}, full, {Target(0,1,16,256,0,4)}, 3},
      {PrimitiveTopology::kTriangleList, {{0,1,2},{3,4,5}}, full, {Target(0,1,16,256,0,4)}, 2},
      {PrimitiveTopology::kTriangleStrip, {{0,1,2},{2,1,3},{2,3,4}}, full, {Target(0,1,16,256,0,4)}, 3},
      {PrimitiveTopology::kTriangleFan, {{0,1,2},{0,2,3},{0,3,4}}, full, {Target(0,1,16,256,0,4)}, 3},
      // Interleaved holes, nonzero source component, separate buffers.
      {PrimitiveTopology::kTriangleList, {{0,1,2},{3,4,5}},
       {{1,2,0,1,0},{0,1,0,4,0},{3,1,1,2,0}},
       {Target(0,1,12,320,24,6),Target(1,2,8,320,12,3)},2},
      // One target fits only one complete triangle. No partial second
      // primitive may write the other, much larger target.
      {PrimitiveTopology::kTriangleList, {{0,1,2},{3,4,5}},
       {{0,4,0,0,0},{0,4,1,0,0}},
       {Target(0,1,16,256,0,4),Target(1,2,16,80,16,4)},1},
      {PrimitiveTopology::kTriangleList, {{0,1,2}}, full,{Target(0,1,16,47,0,4)},0},
      {PrimitiveTopology::kPoints, {{0},{1}}, full,{Target(0,1,16,32,32,4)},0},
      // Two target objects share the same resource, with separate cursors.
      {PrimitiveTopology::kLines, {{0,1},{1,2}}, {{0,2,0,0,0},{2,2,1,0,0}},
       {Target(0,1,16,128,8,2),Target(1,1,256,128,16,2)},2},
      // Unbound used target causes overflow; present unused target is inert.
      {PrimitiveTopology::kPoints, {{0},{1}}, {{0,4,1,0,0}},{Target(0,1,16,32,0,4)},0},
      // Incomplete API primitives have already produced an empty ref array.
      {PrimitiveTopology::kTriangleList, {}, full,{Target(0,1,16,32,0,4)},0},
    };
    for (unsigned i = 0; i < cases.size(); ++i) {
      const auto pending = Make(pool, memory, cases[i], i + 1);
      Check(input.nb_write(pending.txn), "bounded FIFO accepts one draw");
      Check(!input.nb_write(pending.txn), "full input FIFO applies backpressure");
      sc_core::sc_start(sc_core::sc_time(100, sc_core::SC_US));
      PipelineTxn done; Check(output.nb_read(done) && done.sequence == pending.txn.sequence,
                              "completion FIFO retains sequence identity");
      Verify(pool, memory, pending);
      sc_core::sc_start(sc_core::SC_ZERO_TIME);
    }
    // Hold a completed transaction in the depth-one output while the next
    // input completes, proving output backpressure preserves completion order.
    auto first = Make(pool, memory, cases[0], 101);
    Check(input.nb_write(first.txn), "first backpressure job accepted");
    sc_core::sc_start(sc_core::sc_time(100, sc_core::SC_US));
    auto second_case = cases[0]; second_case.targets[0] = Target(0, 9, 16, 256, 0, 4);
    auto second = Make(pool, memory, second_case, 102);
    Check(input.nb_write(second.txn), "second job accepted with output occupied");
    sc_core::sc_start(sc_core::sc_time(100, sc_core::SC_US));
    PipelineTxn done;
    Check(output.nb_read(done) && done.sequence == 101, "output backpressure retains first job");
    Verify(pool, memory, first);
    sc_core::sc_start(sc_core::sc_time(100, sc_core::SC_US));
    Check(output.nb_read(done) && done.sequence == 102, "consumer event resumes pending output");
    Verify(pool, memory, second);
    sc_core::sc_start(sc_core::SC_ZERO_TIME);
    PipelineState state; state.counters.gs_invocations = 71;
    PipelineTxn pass; pass.state = pool.Allocate(sizeof(state)); StorePipelineState(pool, pass.state, state);
    Check(input.nb_write(pass), "non-SO draw accepted");
    sc_core::sc_start(sc_core::sc_time(1, sc_core::SC_US));
    Check(output.nb_read(done), "non-SO draw passes through");
    const auto after = LoadPipelineState(pool, done.state);
    Check(std::memcmp(&state, &after, sizeof(state)) == 0, "non-SO pipeline remains byte-identical");
    pool.Release(done.state);
    Check(pool.bytes_in_flight() == 0 && pool.allocations() == pool.releases(),
          "all tables, native lanes and nested readbacks are released");
    std::cout << "stream_output_test " << mode << ": PASS " << checks << " checks\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << " after " << checks << " checks\n"; return 1;
  }
  return 0;
}
