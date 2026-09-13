// SPDX-License-Identifier: MIT
// Focused GL_KHR_robust_buffer_access_behavior boundary coverage for the
// SystemC VDM -> VertexFetch path. Dynamic vertex-address failures are robust
// reads; malformed binding and EBO transport remain fail-closed elsewhere.
#include "common/functional_types.h"
#include "common/pipeline_state.h"
#include "geometry/vdm.h"
#include "geometry/vertex_fetch.h"

#include <systemc>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace pvrgpu::stub;

enum class Mode {
  kDirect,
  kIndexedNegative,
  kIndexedWrap,
  kRejectIndexRange,
};

unsigned checks = 0;

void Check(bool condition, const char *reason) {
  ++checks;
  if (!condition)
    throw std::runtime_error(reason);
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

Mode ParseMode(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "direct")
    return Mode::kDirect;
  if (argc == 2 && std::string(argv[1]) == "indexed-negative")
    return Mode::kIndexedNegative;
  if (argc == 2 && std::string(argv[1]) == "indexed-wrap")
    return Mode::kIndexedWrap;
  if (argc == 2 && std::string(argv[1]) == "reject-index-range")
    return Mode::kRejectIndexRange;
  throw std::runtime_error(
      "usage: robust-vertex-fetch-test "
      "[direct|indexed-negative|indexed-wrap|reject-index-range]");
}

PipelineState MakeState(MemoryPool &pool, Mode mode) {
  PipelineState state;
  state.width = state.height = 4;
  state.functional_case = FunctionalCase::kDriverPcoTriangles;
  state.stage = PipelineStage::kSubmitted;
  state.draw.topology = PrimitiveTopology::kTriangleList;
  state.drawlist_stats =
      StoreNewArray(pool, std::vector<DrawListStats>{{}});

  // One complete float4 followed by only half of the next element. This is a
  // valid VBO/binding description; vertices 1 and 2 exercise partial and
  // complete dynamic OOB reads without padding the allocation.
  const std::vector<std::uint32_t> words = {
      FloatBits(1.0F), FloatBits(2.0F), FloatBits(3.0F), FloatBits(4.0F),
      FloatBits(5.0F), FloatBits(6.0F),
  };
  VertexBufferResource resource;
  resource.data = StoreNewArray(pool, words);
  resource.byte_size =
      static_cast<std::uint32_t>(words.size() * sizeof(words.front()));
  state.vertex_buffer_resources = StoreNewArray(
      pool, std::vector<VertexBufferResource>{resource});

  VertexAttributeBinding binding;
  binding.stride_bytes = 4U * sizeof(float);
  binding.component_type = VertexComponentType::kFloat32;
  binding.source_components = 4;
  binding.destination_components = 4;
  VertexAttributeBinding oob_binding = binding;
  oob_binding.offset_bytes = resource.byte_size + sizeof(float);
  oob_binding.destination_register = 4;
  state.vertex_attribute_bindings = StoreNewArray(
      pool, std::vector<VertexAttributeBinding>{binding, oob_binding});

  if (mode == Mode::kDirect) {
    state.draw.vertex_count = 3;
    return state;
  }

  state.draw.index_format = IndexFormat::kUint32;
  state.draw.index_count = 3;
  if (mode == Mode::kIndexedNegative) {
    state.draw.base_vertex = -1;
    state.vertex_indices =
        StoreNewArray(pool, std::vector<std::uint32_t>{0, 1, 2});
  } else if (mode == Mode::kIndexedWrap) {
    state.draw.base_vertex = 1;
    state.vertex_indices = StoreNewArray(
        pool, std::vector<std::uint32_t>{UINT32_MAX, 0, 1});
  } else {
    // first_index + index_count exceeds this valid typed payload. Unlike a
    // vertex address outside a VBO, this is malformed captured command data.
    state.draw.first_index = 1;
    state.vertex_indices =
        StoreNewArray(pool, std::vector<std::uint32_t>{0, 1, 2});
  }
  return state;
}

void CheckLane(const VertexLane &lane, int element) {
  std::array<std::uint32_t, 4> expected{};
  if (element == 0) {
    expected = {FloatBits(1.0F), FloatBits(2.0F), FloatBits(3.0F),
                FloatBits(4.0F)};
  } else if (element == 1) {
    expected = {FloatBits(5.0F), FloatBits(6.0F), 0, 0};
  }
  Check(std::equal(expected.begin(), expected.end(), lane.vertex_input),
        "vertex lane did not preserve in-range components and zero OOB ones");
  const std::array<std::uint32_t, 4> robust_offset_zero{};
  Check(std::equal(robust_offset_zero.begin(), robust_offset_zero.end(),
                   lane.vertex_input + 4),
        "a binding whose first element starts beyond its VBO did not zero");
}

}  // namespace

int sc_main(int argc, char **argv) {
  try {
    const Mode mode = ParseMode(argc, argv);
    MemoryPool pool;
    const PipelineState initial = MakeState(pool, mode);
    const PoolHandle state_handle = pool.Allocate(sizeof(PipelineState));
    StorePipelineState(pool, state_handle, initial);

    sc_core::sc_fifo<PipelineTxn> input("input", 1);
    sc_core::sc_fifo<PipelineTxn> middle("middle", 1);
    sc_core::sc_fifo<PipelineTxn> output("output", 1);
    Vdm vdm("vdm", pool);
    VertexFetch fetch("vertex_fetch", pool);
    vdm.input(input);
    vdm.output(middle);
    fetch.input(middle);
    fetch.output(output);
    input.write({state_handle, 1, 1});

    if (mode == Mode::kRejectIndexRange) {
      bool rejected = false;
      try {
        sc_core::sc_start(sc_core::sc_time(20, sc_core::SC_US));
      } catch (const std::exception &error) {
        rejected = std::string(error.what()).find(
                       "indexed draw range exceeds its index buffer") !=
                   std::string::npos;
      }
      Check(rejected,
            "malformed EBO occurrence range was not rejected fail-closed");
      ReleaseFunctionalPayloads(pool, initial);
      pool.Release(state_handle);
      Check(pool.allocations() == pool.releases(),
            "rejected transaction leaked MemoryPool handles");
      std::cout << "robust-vertex-fetch-test: " << checks
                << " checks PASS\n";
      return 0;
    }

    sc_core::sc_start(sc_core::sc_time(20, sc_core::SC_US));
    PipelineTxn completion;
    Check(output.nb_read(completion), "robust vertex fetch did not complete");
    const PipelineState result =
        LoadPipelineState(pool, completion.state);
    Check(result.stage == PipelineStage::kVertexFetched,
          "robust vertex fetch did not reach its completion stage");
    Check(result.counters.ia_vertices == 3 &&
              result.counters.ia_primitives == 1 &&
              result.counters.vs_invocations == 3 &&
              result.counters.vertex_attribute_fetches == 6 &&
              result.counters.vertex_attribute_bytes == 96,
          "robust reads changed input-assembler or vertex-fetch work");

    const auto lanes = LoadArray<VertexLane>(pool, result.vertex_lanes);
    const auto refs = LoadArray<VertexLaneRef>(pool, result.vertex_lane_refs);
    Check(lanes.size() == 3 && refs.size() == 3,
          "robust fetch lost lanes or occurrence references");

    if (mode == Mode::kIndexedNegative) {
      Check(refs[0].vertex_index == UINT32_MAX &&
                refs[1].vertex_index == 0 && refs[2].vertex_index == 1,
            "negative effective index lost its non-addressing lane identity");
      CheckLane(lanes[refs[0].lane_index], -1);
      CheckLane(lanes[refs[1].lane_index], 0);
      CheckLane(lanes[refs[2].lane_index], 1);
    } else {
      // Direct 0,1,2 and indexed UINT32_MAX+1,0+1,1+1 both resolve to the
      // same 32-bit effective sequence. The latter proves required unsigned
      // baseVertex overflow wrapping instead of aborting or saturating.
      for (std::uint32_t index = 0; index < 3; ++index) {
        Check(refs[index].vertex_index == index,
              "direct/wrapped effective vertex identity mismatch");
        CheckLane(lanes[refs[index].lane_index], static_cast<int>(index));
      }
    }

    ReleaseFunctionalPayloads(pool, result);
    pool.Release(state_handle);
    Check(pool.allocations() == pool.releases(),
          "completed transaction leaked MemoryPool handles");
    std::cout << "robust-vertex-fetch-test: " << checks << " checks PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "robust-vertex-fetch-test: " << error.what() << '\n';
    return 1;
  }
}
