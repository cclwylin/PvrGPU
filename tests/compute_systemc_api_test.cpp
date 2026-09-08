#include "pvrgpu_systemc_compute_api.h"
#include "pco_compute_fixtures.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

void Check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

pvrgpu_systemc_compute_dispatch Fixture(unsigned kind, unsigned mode) {
  pvrgpu_systemc_compute_dispatch dispatch{};
  dispatch.version = PVRGPU_SYSTEMC_COMPUTE_API_VERSION;
  const auto &binary = pvrgpu::stub::ComputePcoFixture(kind);
  dispatch.binary = binary.data();
  dispatch.binary_size = binary.size();
  dispatch.memory_mode = mode;
  dispatch.grid[0] = dispatch.grid[1] = dispatch.grid[2] = 1;
  const std::uint32_t local[] = {3, 2, 5};
  for (unsigned i = 0; i < 3; ++i)
    dispatch.block[i] = dispatch.abi.local_size[i] = local[i];
  if (kind == 0)
    return dispatch;
  auto &abi = dispatch.abi;
  abi.stage.temps = kind == 3 ? 10U : kind == 6 ? 5U : 6U;
  abi.stage.vertex_inputs = 1;
  abi.local_invocation_index_count = 1;
  abi.storage_buffer_descriptor_count = 2;
  abi.storage_buffer_used_mask = (kind == 1 || kind == 5 || kind == 6) ? 3U : 2U;
  abi.storage_buffer_read_mask = (kind == 1 || kind == 5) ? 1U : 0U;
  abi.storage_buffer_write_mask = 2U;
  abi.stage.shareds = abi.stage.push_constant_start = 8;
  if (kind == 2) {
    abi.stage.uniform_buffer_descriptor_count = 3;
    abi.uniform_buffer_used_mask = 4;
    abi.storage_buffer_descriptor_start = 12;
    abi.stage.shareds = abi.stage.push_constant_start = 20;
  } else if (kind == 3) {
    abi.stage.coefficients = 7;
    abi.workgroup_id_count = abi.num_workgroups_count = 3;
    abi.num_workgroups_start = 4;
  } else if (kind == 4) {
    abi.stage.shareds = 16;
    abi.stage.push_constant_count = 8;
  } else if (kind == 10 || kind == 11) {
    // Native compiler fixtures: atomicAdd SSBO0[0], then store the old value
    // to SSBO1[local]. Fixture 11 adds CB0 DWORD4 + local instead of one.
    abi.storage_buffer_used_mask = 3;
    abi.storage_buffer_read_mask = 1;
    abi.storage_buffer_write_mask = 3;
    if (kind == 11) {
      abi.stage.shareds = 16;
      abi.stage.push_constant_count = 8;
    }
  } else if (kind >= 13 && kind <= 21) {
    // Native compiler fixtures 13..20: xchg, umin, imin, umax, imax,
    // and, or, xor. SSBO1[local] supplies the operand and receives oldValue.
    // Fixture 21 is the real usclib mutex/LD/ST cmpxchg binary, not DMA AMO:
    // compare=SSBO1[local], replacement=compare+13 (uint32 wrap).
    abi.stage.temps = kind == 21 ? 10U : 7U;
    abi.storage_buffer_used_mask = abi.storage_buffer_read_mask = 3;
    abi.storage_buffer_write_mask = 3;
  } else if (kind == 22 || kind == 23) {
    abi.stage.temps = kind == 22 ? 13U : 21U;
    abi.storage_buffer_used_mask = 3;
    abi.storage_buffer_read_mask = 1;
  }
  return dispatch;
}

pvrgpu_systemc_compute_stats Run(pvrgpu_systemc_compute_dispatch &dispatch) {
  pvrgpu_systemc_compute_stats stats{};
  std::array<char, 512> error{};
  const int status = pvrgpu_systemc_submit_compute(
      &dispatch, &stats, error.data(), error.size());
  Check(status == 0, "native compute dispatch failed: " + std::string(error.data()));
  Check(stats.pool_allocations != 0 &&
            stats.pool_allocations == stats.pool_releases,
        "compute MemoryPool ownership is unbalanced");
  return stats;
}

void Reject(pvrgpu_systemc_compute_dispatch &dispatch, const char *reason) {
  pvrgpu_systemc_compute_stats stats{};
  std::array<char, 512> error{};
  Check(pvrgpu_systemc_submit_compute(&dispatch, &stats, error.data(),
                                       error.size()) != 0 && error[0] != '\0',
        std::string("invalid compute input was accepted: ") + reason);
}

void GuardedVersion() {
#if !defined(_WIN32)
  const long page = sysconf(_SC_PAGESIZE);
  Check(page > 0, "guard-page size unavailable");
  void *mapping = mmap(nullptr, 2U * page, PROT_NONE,
                         MAP_PRIVATE | MAP_ANON, -1, 0);
  Check(mapping != MAP_FAILED &&
            mprotect(mapping, page, PROT_READ | PROT_WRITE) == 0,
        "compute version guard-page allocation failed");
  // Align the envelope as its ABI requires; only the version is initialized.
  constexpr std::size_t readable = alignof(pvrgpu_systemc_compute_dispatch);
  auto *bytes = static_cast<std::uint8_t *>(mapping) + page - readable;
  auto *old = reinterpret_cast<pvrgpu_systemc_compute_dispatch *>(bytes);
  for (const std::uint32_t invalid_version : {0U,1U,2U,3U,4U,5U}) {
    std::memcpy(bytes, &invalid_version, sizeof(invalid_version));
    Reject(*old, "old envelope with inaccessible tail");
  }
  munmap(mapping, 2U * page);
#endif
}

void GuardedOldStats(unsigned mode) {
#if !defined(_WIN32)
  static_assert(PVRGPU_SYSTEMC_COMPUTE_API_VERSION == 6);
  // API v1 had thirteen uint64_t counters. Its caller may allocate exactly
  // that much: rejecting v1 must happen before clearing the larger stats.
  constexpr std::size_t old_stats_size = 13U * sizeof(std::uint64_t);
  static_assert(sizeof(pvrgpu_systemc_compute_stats) ==
                old_stats_size + 3U * sizeof(std::uint64_t));
  const long page = sysconf(_SC_PAGESIZE);
  Check(page > 0 && static_cast<std::size_t>(page) >= old_stats_size,
        "old-stats guard-page size unavailable");
  void *mapping = mmap(nullptr, 2U * page, PROT_NONE,
                      MAP_PRIVATE | MAP_ANON, -1, 0);
  Check(mapping != MAP_FAILED &&
            mprotect(mapping, page, PROT_READ | PROT_WRITE) == 0,
        "old-stats guard-page allocation failed");
  auto *bytes = static_cast<std::uint8_t *>(mapping) + page - old_stats_size;
  std::memset(bytes, 0xa5, old_stats_size);
  auto dispatch = Fixture(0, mode);
  dispatch.version = 1;
  std::array<char, 512> error{};
  const int status = pvrgpu_systemc_submit_compute(
      &dispatch, reinterpret_cast<pvrgpu_systemc_compute_stats *>(bytes),
      error.data(), error.size());
  const bool unchanged = std::all_of(bytes, bytes + old_stats_size,
                                    [](std::uint8_t value) { return value == 0xa5; });
  munmap(mapping, 2U * page);
  Check(status != 0 && error[0] != '\0', "API v1 envelope was accepted by API v2");
  Check(unchanged, "rejecting API v1 modified its old stats buffer");
#else
  (void)mode;
#endif
}

void InvalidEnvelopes(unsigned mode) {
  for (const unsigned bad : {0U,1U,2U,3U,4U,5U,6U}) {
    auto shared = Fixture(0, mode);
    shared.abi.shared_memory_bytes = 4;
    shared.abi.shared_memory_descriptor_count = 4;
    shared.abi.stage.shareds = shared.abi.stage.push_constant_start = 4;
    if (bad == 0) shared.abi.shared_memory_bytes = 32772;
    if (bad == 1) shared.abi.shared_memory_bytes = 3;
    if (bad == 2) shared.abi.shared_memory_descriptor_count = 3;
    if (bad == 3) shared.abi.shared_memory_descriptor_start = 1;
    if (bad == 4) shared.abi.stage.push_constant_start = 0;
    if (bad == 5) shared.abi.stage.shareds = 3;
    if (bad == 6) shared.abi.shared_memory_bytes = 0;
    Reject(shared, "private shared descriptor extent/overlap/count/size");
  }
  // A genuine empty CS may reserve exactly the supported workgroup bound.
  // This validates allocation and descriptor transport without host shaders.
  for (const unsigned bytes : {4U,32768U}) {
    auto shared = Fixture(0, mode);
    shared.abi.shared_memory_bytes = bytes;
    shared.abi.shared_memory_descriptor_count = 4;
    shared.abi.stage.shareds = shared.abi.stage.push_constant_start = 4;
    const auto stats = Run(shared);
    Check(stats.workgroups == 1 && stats.invocations == 30 &&
          stats.memory_instructions == 0, "shared legal boundary invented shader accesses");
  }
  auto invalid = Fixture(1, mode);
  Reject(invalid, "shader resource masks reference absent bindings");
  invalid = Fixture(0, mode);
  invalid.abi.storage_buffer_read_mask = 1;
  Reject(invalid, "read mask is not a subset of used mask");
  // Describe deliberately unreadable extents; the aggregate limit must be
  // checked before any payload is copied. Only one real DWORD is needed.
  std::uint32_t sentinel = 0x87654321U;
  std::array<pvrgpu_systemc_compute_resource, 2> resources{{
      {reinterpret_cast<std::uint8_t *>(&sentinel),
       PVRGPU_SYSTEMC_COMPUTE_MAX_RESOURCE_BYTES},
      {reinterpret_cast<std::uint8_t *>(&sentinel), sizeof(sentinel)},
  }};
  invalid = Fixture(0, mode);
  invalid.resources = resources.data();
  invalid.resource_count = resources.size();
  Reject(invalid, "aggregate snapshot bound must precede byte copies");
  invalid = Fixture(0, mode);
  invalid.abi.stage.uniform_buffer_descriptor_count = 1;
  invalid.abi.storage_buffer_descriptor_start = 4;
  invalid.abi.stage.push_constant_start = 4;
  invalid.abi.stage.shareds = 4;
  invalid.abi.uniform_buffer_used_mask = 1;
  pvrgpu_systemc_compute_resource resource{
      reinterpret_cast<std::uint8_t *>(&sentinel), sizeof(sentinel)};
  pvrgpu_systemc_compute_binding view{0, 0, 0, 3, 0, sizeof(sentinel)};
  invalid.resources = &resource;
  invalid.resource_count = 1;
  invalid.bindings = &view;
  invalid.binding_count = 1;
  Reject(invalid, "UBO view cannot acquire write permission");
  view.access = 1;
  view.offset = UINT64_MAX;
  Reject(invalid, "binding offset overflow");
  Check(sentinel == 0x87654321U, "invalid compute input modified caller bytes");
}

void CopyAndAlias(unsigned mode, unsigned kind) {
  auto dispatch = Fixture(kind, mode);
  std::array<std::uint32_t, 80> backing{};
  backing.fill(0xdeadbeefU);
  for (unsigned i = 0; i < 30; ++i)
    backing[4 + i] = 0xfedc0000U + i * 13U;
  pvrgpu_systemc_compute_resource resource{
      reinterpret_cast<std::uint8_t *>(backing.data()), sizeof(backing)};
  std::array<pvrgpu_systemc_compute_binding, 2> views{{
      {kind == 2 ? 0U : 1U, kind == 2 ? 2U : 0U, 0, 1, 16, 120},
      {1, 1, 0, 2, 160, 120},
  }};
  dispatch.resources = &resource;
  dispatch.resource_count = 1;
  dispatch.bindings = views.data();
  dispatch.binding_count = views.size();
  for (unsigned iteration = 0; iteration < 2; ++iteration) {
    // Reusing a session and address slot must invalidate the prior SLC data.
    for (unsigned i = 0; i < 30; ++i)
      backing[4 + i] ^= 0x31234567U;
    const auto before = backing;
    const auto stats = Run(dispatch);
    Check(stats.invocations == 30 && stats.workgroups == 1 &&
              stats.atomic_instructions == 0 &&
              stats.load_instructions == 30 && stats.store_instructions == 30 &&
              stats.readback_bytes == sizeof(backing),
          "copy must perform native per-lane LD/ST and raw BO readback");
    for (unsigned i = 0; i < backing.size(); ++i) {
      const auto expected = i >= 40 && i < 70 ? before[4 + i - 40] : before[i];
      Check(backing[i] == expected, "aliased BO view copy changed an unexpected word");
    }
    if (mode == 0)
      Check(stats.dram_read_bytes == 0 && stats.dram_write_bytes == 0 &&
                stats.direct_write_bytes == 120,
            "direct compute must not claim DRAM traffic");
    else
      Check(stats.dram_read_bytes != 0 && stats.dram_write_bytes != 0 &&
                stats.direct_read_bytes == 0 && stats.direct_write_bytes == 0,
            "modeled compute must report real DRAM traffic");
  }

  // A late lane fault may change model DRAM, but must not partially publish a
  // failed dispatch into the caller's BO. Completion must still drain FIFOs.
  const auto before = backing;
  views[1].bytes_size = 116;
  Reject(dispatch, "last-lane output exceeds its writable view");
  Check(backing == before, "failed compute dispatch partially committed host bytes");
  views[1].bytes_size = 120;
  (void)Run(dispatch);
}

void WideVectorCopy(unsigned mode, unsigned kind) {
  const unsigned width = kind == 22 ? 8U : 16U;
  const unsigned words = 30U * width;
  constexpr unsigned input_start = 16;
  const unsigned output_start = input_start + words + 16U;
  std::vector<std::uint32_t> backing(output_start + words + 16U, 0xa5a5a5a5U);
  pvrgpu_systemc_compute_resource resource{
      reinterpret_cast<std::uint8_t *>(backing.data()), backing.size() * 4U};
  std::array<pvrgpu_systemc_compute_binding, 2> views{{
      {1, 0, 0, 1, input_start * 4U, words * 4U},
      {1, 1, 0, 2, output_start * 4U, words * 4U},
  }};
  auto dispatch = Fixture(kind, mode);
  dispatch.resources = &resource;
  dispatch.resource_count = 1;
  dispatch.bindings = views.data();
  dispatch.binding_count = views.size();
  for (unsigned iteration = 0; iteration < 2; ++iteration) {
    for (unsigned i = 0; i < words; ++i)
      backing[input_start + i] = 0xffff0000U ^ (i * 137U + iteration * 991U);
    const auto before = backing;
    const auto stats = Run(dispatch);
    Check(stats.invocations == 30 && stats.load_instructions == 30 &&
              stats.store_instructions == 30 && stats.atomic_instructions == 0 &&
              stats.readback_bytes == resource.bytes_size,
          "wide-vector copy must execute true native LD/ST for each active lane");
    for (unsigned i = 0; i < backing.size(); ++i) {
      const auto expected = i >= output_start && i < output_start + words
          ? before[input_start + i - output_start] : before[i];
      Check(backing[i] == expected, "wide-vector copy corrupted data or a guard word");
    }
    views[1].bytes_size -= 4U;
    const auto guarded = backing;
    Reject(dispatch, "wide-vector final component exceeds writable view");
    Check(backing == guarded, "failed vector access published partial host bytes");
    views[1].bytes_size += 4U;
  }
  (void)Run(dispatch); // Recover cleanly after the final out-of-range vector.
}

void BuiltinsAndMetadata(unsigned mode) {
  std::array<std::uint32_t, 120> output{};
  output.fill(0xaabbccddU);
  pvrgpu_systemc_compute_resource resource{
      reinterpret_cast<std::uint8_t *>(output.data()), sizeof(output)};
  pvrgpu_systemc_compute_binding view{1, 1, 0, 2, 0, sizeof(output)};
  auto dispatch = Fixture(3, mode);
  dispatch.resources = &resource;
  dispatch.resource_count = 1;
  dispatch.bindings = &view;
  dispatch.binding_count = 1;
  const auto ids = Run(dispatch);
  Check(ids.invocations == 30 && ids.store_instructions == 30,
        "builtin fixture did not run one native vector store per lane");
  for (unsigned index = 0; index < 30; ++index) {
    const unsigned x = index % 3U, y = (index / 3U) % 2U, z = index / 6U;
    Check(output[4 * index] == 2U * x + 1U &&
              output[4 * index + 1] == 2U * y + 1U &&
              output[4 * index + 2] == 2U * z + 1U &&
              output[4 * index + 3] == 0xaabbccddU,
          "native non-power-of-two builtin lowering produced incorrect IDs");
  }
  dispatch = Fixture(6, mode);
  std::array<pvrgpu_systemc_compute_binding, 2> views{{
      {1, 0, 0, 0, 256, 60}, {1, 1, 0, 2, 0, 120},
  }};
  dispatch.resources = &resource;
  dispatch.resource_count = 1;
  dispatch.bindings = views.data();
  dispatch.binding_count = views.size();
  const auto length = Run(dispatch);
  Check(length.load_instructions == 0 && length.store_instructions == 30,
        "buffer-size descriptor query must not synthesize a memory load");
  for (unsigned index = 0; index < 30; ++index)
    Check(output[index] == 60, "SSBO size query ignored its bound view extent");
  dispatch = Fixture(4, mode);
  std::array<std::uint32_t, 8> push{};
  push[4] = 137;
  dispatch.push_words = push.data();
  dispatch.push_word_count = push.size();
  dispatch.resources = &resource;
  dispatch.resource_count = 1;
  dispatch.bindings = &view;
  dispatch.binding_count = 1;
  (void)Run(dispatch);
  for (unsigned index = 0; index < 30; ++index)
    Check(output[index] == 137 + index, "CB0 push data was not relocated after descriptors");
}

void AtomicAddAndAlias(unsigned mode, unsigned kind) {
  constexpr unsigned lanes = 30;
  for (unsigned layout = 0; layout < 3; ++layout) {
    auto dispatch = Fixture(kind, mode);
    // Separate BOs, disjoint same-BO views, and overlapping same-BO views.
    // The latter must not be snapshotted as two independent copies: publishing
    // the output view must preserve the update visible through the counter.
    std::vector<std::uint32_t> backing0(layout ? 80 : 8, 0xdeadbeefU);
    std::vector<std::uint32_t> backing1(layout ? 0 : 40, 0xaabbccddU);
    std::array<pvrgpu_systemc_compute_resource, 2> resources{{
        {reinterpret_cast<std::uint8_t *>(backing0.data()), backing0.size() * 4U},
        {reinterpret_cast<std::uint8_t *>(backing1.data()), backing1.size() * 4U},
    }};
    const auto counter_view_size = layout == 2 ? resources[0].bytes_size - 16U : 4U;
    std::array<pvrgpu_systemc_compute_binding, 2> views{{
        {1, 0, 0, 3, 16, counter_view_size},
        {1, 1, layout ? 0U : 1U, 2, layout ? 160U : 16U, lanes * 4U},
    }};
    std::array<std::uint32_t, 8> push{};
    if (kind == 11) {
      dispatch.push_words = push.data();
      dispatch.push_word_count = push.size();
    }
    dispatch.resources = resources.data();
    dispatch.resource_count = layout ? 1 : 2;
    dispatch.bindings = views.data();
    dispatch.binding_count = views.size();
    auto *counter = &backing0[4];
    auto *old_values = layout ? &backing0[40] : &backing1[4];
    const auto run_and_verify = [&]() {
      const auto before0 = backing0;
      const auto before1 = backing1;
      const std::uint32_t initial = *counter;
      std::uint32_t expected_counter = initial;
      for (unsigned lane = 0; lane < lanes; ++lane)
        expected_counter += kind == 11 ? push[4] + lane : 1U;
      const auto stats = Run(dispatch);
      Check(stats.workgroups == 1 && stats.invocations == lanes &&
                stats.atomic_instructions == lanes &&
                stats.load_instructions == 0 && stats.store_instructions == lanes &&
                stats.memory_instructions >= 2U * lanes,
            "atomic fixture must execute native per-lane RMW and output ST");
      Check(*counter == expected_counter, "native atomic add lost updates or uint32 wrap");

      // Check linearizability without requiring any particular lane order.
      // Each returned old value must begin one unused lane's update, forming
      // exactly one chain from the initial word to the final counter word.
      std::array<bool, lanes> visited{};
      std::uint32_t cursor = initial;
      for (unsigned step = 0; step < lanes; ++step) {
        unsigned found = lanes;
        for (unsigned lane = 0; lane < lanes; ++lane)
          if (!visited[lane] && old_values[lane] == cursor) {
            Check(found == lanes, "atomic return values duplicate an old counter word");
            found = lane;
          }
        Check(found != lanes, "atomic return values do not form a serialized RMW chain");
        visited[found] = true;
        cursor += kind == 11 ? push[4] + found : 1U;
      }
      Check(cursor == expected_counter, "atomic old values disagree with the final counter");
      for (unsigned index = 0; index < backing0.size(); ++index)
        if (index != 4 && !(layout && index >= 40 && index < 40 + lanes))
          Check(backing0[index] == before0[index], "atomic dispatch changed a BO guard word");
      for (unsigned index = 0; index < backing1.size(); ++index)
        if (index < 4 || index >= 4 + lanes)
          Check(backing1[index] == before1[index], "atomic output changed a BO guard word");
      const auto readback = resources[0].bytes_size +
                            (layout ? 0U : resources[1].bytes_size);
      Check(stats.readback_bytes == readback, "atomic alias readback counted a BO twice");
      if (mode == 0) {
        Check(stats.direct_read_bytes == lanes * 4U + readback &&
                  stats.direct_write_bytes == lanes * 8U &&
                  stats.dram_read_bytes == 0 && stats.dram_write_bytes == 0,
              "direct atomic traffic must count real RMW, store and BO readback bytes");
      } else if (mode == 1) {
        Check(stats.dram_read_bytes == lanes * 4U + readback &&
                  stats.dram_write_bytes == lanes * 8U &&
                  stats.direct_read_bytes == 0 && stats.direct_write_bytes == 0,
              "bypass atomic traffic must count actual DRAM reads and writes");
      } else {
        Check(stats.dram_read_bytes != 0 && stats.dram_write_bytes != 0 &&
                  stats.direct_read_bytes == 0 && stats.direct_write_bytes == 0,
              "cached atomic traffic must use the modeled coherent memory service");
      }
    };
    for (unsigned iteration = 0; iteration < 2; ++iteration) {
      *counter = iteration ? 0x10203040U : UINT32_MAX - 11U;
      push[4] = 17U + iteration;
      run_and_verify();
    }

    // Restrict the counter view before narrowing the output, so the broad RW
    // alias cannot legitimately grant the final output store another view.
    views[0].bytes_size = 4;
    views[1].bytes_size = (lanes - 1U) * 4U;
    const auto before0 = backing0;
    const auto before1 = backing1;
    Reject(dispatch, "last atomic result store exceeds every writable view");
    Check(backing0 == before0 && backing1 == before1,
          "late atomic fault partially committed counter or output BO bytes");
    views[0].bytes_size = counter_view_size;
    views[1].bytes_size = lanes * 4U;
    // Reusing the same session proves failed RMW/output requests and pool
    // payloads were drained; fresh host input replaces any dirty model bytes.
    run_and_verify();
  }
}

std::uint32_t AtomicOperation(unsigned kind, std::uint32_t old,
                              std::uint32_t operand) {
  switch (kind) {
  case 13: return operand;
  case 14: return std::min(old, operand);
  case 15: return (old ^ 0x80000000U) < (operand ^ 0x80000000U) ? old : operand;
  case 16: return std::max(old, operand);
  case 17: return (old ^ 0x80000000U) > (operand ^ 0x80000000U) ? old : operand;
  case 18: return old & operand;
  case 19: return old | operand;
  case 20: return old ^ operand;
  case 21: return old == operand ? operand + 13U : old;
  default: throw std::runtime_error("unknown atomic test operation");
  }
}

void CheckAtomicSerialization(unsigned kind, std::uint32_t initial,
                              std::uint32_t final,
                              const std::array<std::uint32_t, 30> &operands,
                              const std::uint32_t *old_values) {
  // Unlike add-by-one, extrema/bitwise/CAS produce legitimate duplicate old
  // values and self-loops. Check a complete Euler trail through every observed
  // old->new transition, without requiring the model's lane issue order.
  constexpr unsigned lanes = 30;
  std::array<std::uint32_t, lanes> next{};
  std::array<bool, lanes> used{};
  std::array<std::uint32_t, lanes + 1> stack{}, reverse{};
  for (unsigned lane = 0; lane < lanes; ++lane)
    next[lane] = AtomicOperation(kind, old_values[lane], operands[lane]);
  unsigned top = 1, length = 0, consumed = 0;
  stack[0] = initial;
  while (top) {
    unsigned edge = 0;
    while (edge < lanes && (used[edge] || old_values[edge] != stack[top - 1]))
      ++edge;
    if (edge == lanes) {
      reverse[length++] = stack[--top];
    } else {
      used[edge] = true;
      ++consumed;
      stack[top++] = next[edge];
    }
  }
  Check(consumed == lanes && length == lanes + 1 &&
            reverse[0] == final && reverse[lanes] == initial,
        "atomic return-old graph does not connect initial to final value");
  used.fill(false);
  for (unsigned index = lanes; index > 0; --index) {
    unsigned edge = 0;
    while (edge < lanes && (used[edge] || old_values[edge] != reverse[index] ||
                            next[edge] != reverse[index - 1]))
      ++edge;
    Check(edge != lanes, "atomic return-old graph is not a complete legal serialization");
    used[edge] = true;
  }
}

void AtomicOperationsAndAlias(unsigned mode, unsigned kind) {
  constexpr unsigned lanes = 30; // The last two physical task lanes are masked.
  for (unsigned layout = 0; layout < 3; ++layout) {
    auto dispatch = Fixture(kind, mode);
    std::vector<std::uint32_t> backing0(layout ? 88 : 16, 0xdeadbeefU);
    std::vector<std::uint32_t> backing1(layout ? 0 : 48, 0xaabbccddU);
    std::array<pvrgpu_systemc_compute_resource, 2> resources{{
        {reinterpret_cast<std::uint8_t *>(backing0.data()), backing0.size() * 4U},
        {reinterpret_cast<std::uint8_t *>(backing1.data()), backing1.size() * 4U},
    }};
    std::array<pvrgpu_systemc_compute_binding, 2> views{{
        {1, 0, 0, 3, 16, 4},
        {1, 1, layout ? 0U : 1U, 3, layout ? 160U : 16U, lanes * 4U},
    }};
    dispatch.resources = resources.data();
    dispatch.resource_count = layout ? 1 : 2;
    dispatch.bindings = views.data();
    dispatch.binding_count = views.size();
    const auto run_and_verify = [&](std::uint32_t initial, unsigned iteration) {
      // Rebind both offsets in the same session, not just their host contents.
      // A stale descriptor or SLC line would touch a checked guard word.
      const unsigned counter_index = iteration & 1U ? 8U : 4U;
      const unsigned output_index = (layout ? 40U : 4U) + (iteration & 1U) * 4U;
      views[0].offset = counter_index * 4U;
      views[0].bytes_size = layout == 2 ? resources[0].bytes_size - views[0].offset : 4U;
      views[1].offset = output_index * 4U;
      views[1].bytes_size = lanes * 4U;
      auto *counter = &backing0[counter_index];
      auto *old_values = layout ? &backing0[output_index] : &backing1[output_index];
      *counter = initial;
      std::array<std::uint32_t, lanes> operands{};
      static constexpr std::uint32_t extrema[] = {
          0U, 1U, 0xffffffffU, 0x7fffffffU, 0x80000000U, 0x80000001U,
          0xaaaaaaaaU, 0x55555555U, 0x12345678U, 0xfedcba98U};
      for (unsigned lane = 0; lane < lanes; ++lane) {
        operands[lane] = kind == 21 ? initial + (lane / 2U) * 13U
                                    : extrema[(lane + iteration) % 10U];
        old_values[lane] = operands[lane];
      }
      const auto before0 = backing0;
      const auto before1 = backing1;
      const auto stats = Run(dispatch);
      Check(stats.workgroups == 1 && stats.invocations == lanes,
            "atomic operation did not preserve the partial task lane mask");
      CheckAtomicSerialization(kind, initial, *counter, operands, old_values);
      if (kind != 21) {
        Check(stats.atomic_instructions == lanes &&
                  stats.load_instructions == lanes && stats.store_instructions == lanes,
              "DMA atomic operation must count operand LD, AMO, and old-value ST per active lane");
        if (kind != 13) {
          std::uint32_t expected = initial;
          for (auto operand : operands)
            expected = AtomicOperation(kind, expected, operand);
          Check(*counter == expected, "commutative atomic final value differs from its operands");
        }
      } else {
        unsigned successes = 0;
        for (unsigned lane = 0; lane < lanes; ++lane)
          successes += old_values[lane] == operands[lane];
        Check(successes > 0 && successes < lanes,
              "native CAS fixture must exercise both replacement and failed comparison");
        Check(stats.atomic_instructions == 0 && stats.load_instructions >= 2U * lanes &&
                  stats.store_instructions >= lanes + successes,
              "native usclib CAS must use mutex-protected LD/ST, not invented DMA AMO counts");
      }
      for (unsigned index = 0; index < backing0.size(); ++index)
        if (index != counter_index && !(layout && index >= output_index && index < output_index + lanes))
          Check(backing0[index] == before0[index], "atomic operation changed a counter/alias guard word");
      for (unsigned index = 0; index < backing1.size(); ++index)
        if (index < output_index || index >= output_index + lanes)
          Check(backing1[index] == before1[index], "atomic operation changed an output guard word");
      const auto readback = resources[0].bytes_size + (layout ? 0U : resources[1].bytes_size);
      Check(stats.readback_bytes == readback, "atomic alias resource was not read back exactly once");
      if (mode == 0) {
        Check(stats.direct_read_bytes >= lanes * 8U + readback &&
                  stats.direct_write_bytes >= lanes * 4U &&
                  stats.dram_read_bytes == 0 && stats.dram_write_bytes == 0,
              "direct atomic path must report actual LD/RMW/ST and BO readback");
      } else {
        Check(stats.dram_read_bytes != 0 && stats.dram_write_bytes != 0 &&
                  stats.direct_read_bytes == 0 && stats.direct_write_bytes == 0,
              "atomic path must traverse selected modeled memory mode");
      }
    };
    run_and_verify(0xffffffffU, 0);
    run_and_verify(0x80000000U, 1);
    run_and_verify(0x7fffffffU, 2);

    // The broad alias must not hide an intentionally insufficient output view.
    views[0].bytes_size = 4;
    views[1].bytes_size = (lanes - 1U) * 4U;
    auto before0 = backing0;
    auto before1 = backing1;
    Reject(dispatch, "atomic operand/result exceeds every allowed view");
    Check(backing0 == before0 && backing1 == before1,
          "failed atomic operation partially published caller BO bytes");
    views[1].bytes_size = lanes * 4U;
    // RMW requires one view with both rights; read-only or write-only views
    // cannot silently turn it into a plain LD/ST or acquire missing access.
    for (unsigned access : {1U, 2U}) {
      views[0].access = access;
      Reject(dispatch, "atomic counter view lacks combined read/write permission");
      Check(backing0 == before0 && backing1 == before1,
            "rejected atomic access mask modified caller bytes");
    }
    views[0].access = 3;
    run_and_verify(0xfffffff8U, 3); // Also covers CAS replacement uint32 wrap.
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const unsigned mode = argc > 1 ? static_cast<unsigned>(std::stoul(argv[1])) : 0;
    GuardedVersion();
    GuardedOldStats(mode);
    InvalidEnvelopes(mode);
    auto invalid = Fixture(0, mode);
    invalid.abi.stage.shareds = 257;
    Reject(invalid, "oversized shared-register file");
    invalid = Fixture(0, mode);
    invalid.block[0] = 4;
    Reject(invalid, "runtime local size differs from compiled size");
    invalid = Fixture(0, mode);
    invalid.abi.stage.entry_offset = 1;
    Reject(invalid, "unsupported nonzero compute entry point");
    auto empty = Fixture(0, mode);
    const auto result = Run(empty);
    Check(result.invocations == 30 && result.workgroups == 1 &&
              result.atomic_instructions == 0 &&
              result.load_instructions == 0 && result.store_instructions == 0,
          "empty compute shader lacks native task completion");
    empty.grid[0] = 0;
    Check(Run(empty).invocations == 0, "zero-grid dispatch ran an invocation");
    for (unsigned kind : {1U, 2U, 5U})
      CopyAndAlias(mode, kind);
    BuiltinsAndMetadata(mode);
    for (unsigned kind : {10U, 11U})
      AtomicAddAndAlias(mode, kind);
    for (unsigned kind = 13; kind <= 21; ++kind)
      AtomicOperationsAndAlias(mode, kind);
    for (unsigned kind : {22U, 23U})
      WideVectorCopy(mode, kind);
    std::cout << "compute SystemC API mode=" << mode << ": PASS\n";
    return 0;
  } catch (const std::exception &failure) {
    std::cerr << "compute SystemC API: " << failure.what() << '\n';
    return 1;
  }
}
