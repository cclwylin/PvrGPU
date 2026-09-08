// SPDX-License-Identifier: MIT
#include "gallium/drivers/pvrgpu/pvrgpu_indirect_draw.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

static unsigned checks;
static void Check(bool condition, const char *reason) {
  ++checks;
  if (!condition) {
    std::fprintf(stderr, "indirect-draw-test: %s\n", reason);
    std::exit(1);
  }
}

static bool Equal(const pvrgpu_indirect_draw &left,
                  const pvrgpu_indirect_draw &right) {
  return left.count == right.count && left.instance_count == right.instance_count &&
         left.first == right.first && left.base_vertex == right.base_vertex &&
         left.base_instance == right.base_instance;
}

int main() {
  for (unsigned width : {0U, 1U, 2U, 4U}) {
    const std::size_t size = width ? 20 : 16;
    for (std::size_t offset : {0U, 4U, 12U}) {
      for (std::size_t stride : {0U, 20U, 32U}) {
        // Deliberately unaligned backing tests memcpy decoding; only the API
        // offset, not the implementation's allocation, needs 4-byte alignment.
        std::array<uint8_t, 80> storage{};
        storage.fill(0xa5);
        const std::array<uint32_t, 5> command{{4, 3, 7, width ? 2U : 9U, 9}};
        std::memcpy(storage.data() + 1 + offset, command.data(), size);
        const auto original = storage;
        pvrgpu_indirect_draw draw{};
        const char *reason = "unset";
        Check(pvrgpu_decode_indirect_draw(storage.data() + 1, offset + size,
                  offset, stride, 1, width, false, false, &draw, &reason),
              "single command accepts exact span without requiring stride tail");
        Check(Equal(draw, {4, 3, 7, width ? 2 : 0, 9}) && reason == nullptr,
              "count/instances/first/base vertex/base instance preserve their ABI fields");
        Check(!pvrgpu_indirect_draw_is_empty(&draw), "nonempty work detected");
        Check(storage == original, "decoding leaves source and guards untouched");
      }
    }
  }

  const pvrgpu_indirect_draw sentinel{11, 22, 33, -44, 55};
  const std::array<uint32_t, 5> command{{4, 1, 0, 0, 0}};
  const auto Reject = [&](const void *buffer, std::size_t capacity,
                          std::size_t offset, std::size_t stride,
                          unsigned count, unsigned width, bool counted,
                          bool stream, const char *expected) {
    auto draw = sentinel;
    const char *reason = nullptr;
    Check(!pvrgpu_decode_indirect_draw(buffer, capacity, offset, stride,
              count, width, counted, stream, &draw, &reason),
          "unsupported or malformed command rejected");
    Check(reason && std::strcmp(reason, expected) == 0, "precise rejection reason");
    Check(Equal(draw, sentinel), "failure does not publish partial decoded output");
  };
  for (unsigned width : {0U, 1U, 2U, 4U}) {
    for (std::size_t capacity = 0; capacity < (width ? 20U : 16U); ++capacity)
      Reject(command.data(), capacity, 0, 0, 1, width, false, false,
             "indirect_buffer_bounds");
    for (std::size_t offset : {1U, 2U, 3U})
      Reject(command.data(), sizeof(command), offset, 0, 1, width, false, false,
             "indirect_alignment_or_stride");
    for (std::size_t stride : {1U, 2U, 3U, 4U, 12U, 19U})
      Reject(command.data(), sizeof(command), 0, stride, 1, width, false, false,
             "indirect_alignment_or_stride");
    Reject(command.data(), sizeof(command), std::numeric_limits<std::size_t>::max() - 3,
           0, 1, width, false, false, "indirect_buffer_bounds");
    Reject(command.data(), sizeof(command), 24, 0, 1, width, false, false,
           "indirect_buffer_bounds");
  }
  for (unsigned width : {3U, 8U, std::numeric_limits<unsigned>::max()})
    Reject(command.data(), sizeof(command), 0, 0, 1, width, false, false,
           "indirect_index_size");
  for (unsigned count : {0U, 2U, std::numeric_limits<unsigned>::max()})
    Reject(command.data(), sizeof(command), 0, 0, count, 2, false, false,
           "indirect_variant");
  Reject(command.data(), sizeof(command), 0, 0, 1, 2, true, false, "indirect_variant");
  Reject(command.data(), sizeof(command), 0, 0, 1, 2, false, true, "indirect_variant");
  Reject(nullptr, sizeof(command), 0, 0, 1, 2, false, false, "indirect_buffer_missing");
  Check(!pvrgpu_decode_indirect_draw(command.data(), sizeof(command), 0, 0,
            1, 2, false, false, nullptr, nullptr), "missing output rejected without reason pointer");

  for (int32_t base : {INT32_MIN, -17, -1, 0, 1, 17, INT32_MAX}) {
    auto params = command;
    std::memcpy(&params[3], &base, sizeof(base));
    params[4] = UINT32_MAX;
    pvrgpu_indirect_draw draw{};
    Check(pvrgpu_decode_indirect_draw(params.data(), sizeof(params), 0, 0,
              1, 2, false, false, &draw, nullptr), "signed baseVertex is decoded, not unsigned-cast");
    Check(draw.base_vertex == base && draw.base_instance == UINT32_MAX,
          "all signed extremes and unsigned baseInstance preserved");
  }
  for (unsigned width : {0U, 2U}) {
    for (uint32_t count : {0U, 1U, UINT32_MAX}) {
      for (uint32_t instances : {0U, 1U, UINT32_MAX}) {
        if (count && instances) continue;
        std::array<uint32_t, 5> params{{count, instances, UINT32_MAX, UINT32_MAX, UINT32_MAX}};
        pvrgpu_indirect_draw draw{};
        Check(pvrgpu_decode_indirect_draw(params.data(), sizeof(params), 0, 0,
                  1, width, false, false, &draw, nullptr), "zero-work command does not demand vertex buffers");
        Check(pvrgpu_indirect_draw_is_empty(&draw), "zero count OR zero instances is truly empty");
      }
    }
    std::array<uint32_t, 5> first_wrap{{2, 1, UINT32_MAX, 0, 0}};
    Reject(first_wrap.data(), sizeof(first_wrap), 0, 0, 1, width, false, false,
           "indirect_vertex_or_instance_range");
    std::array<uint32_t, 5> instance_wrap{{1, 2, 0, UINT32_MAX, UINT32_MAX}};
    Reject(instance_wrap.data(), sizeof(instance_wrap), 0, 0, 1, width, false, false,
           "indirect_vertex_or_instance_range");
  }
  Check(!pvrgpu_indirect_draw_is_empty(nullptr), "missing command is not a successful no-op");

  // The snapshot owns values, not pointers into argument storage. A completed
  // producer's next update is visible to the next decode without retroactively
  // changing a draw already decoded. Actual TF/CS completion ordering is tested
  // by the native indirect integration probes, not simulated by this unit.
  auto resource = command;
  pvrgpu_indirect_draw first{}, next{};
  Check(pvrgpu_decode_indirect_draw(resource.data(), sizeof(resource), 0, 0,
            1, 2, false, false, &first, nullptr), "first argument snapshot");
  resource = {{7, 2, 3, 4, 5}};
  Check(pvrgpu_decode_indirect_draw(resource.data(), sizeof(resource), 0, 0,
            1, 2, false, false, &next, nullptr), "next snapshot sees completed producer bytes");
  Check(Equal(first, {4, 1, 0, 0, 0}) && Equal(next, {7, 2, 3, 4, 5}),
        "decoded command is independent of later resource writes");
  std::printf("indirect-draw-test: %u checks PASS\n", checks);
}
