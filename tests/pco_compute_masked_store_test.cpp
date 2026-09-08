/* SPDX-License-Identifier: MIT */
// Execute freshly compiled native PCO, not reconstructed store instructions.
// Observe each memory transaction so a read/modify/write cannot pass merely
// because it happens to restore the unselected component's original value.
#include "shader/compute_iss.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

using namespace pvrgpu::stub;
namespace {
std::uint64_t checks = 0;
void Check(bool condition, const char *message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}

ComputePcoAbi ReadAbi(const std::string &path) {
  auto *file = std::fopen(path.c_str(), "r");
  Check(file != nullptr, "cannot open compiled ABI");
  ComputePcoAbi a;
  const auto count = std::fscanf(file,
      "temps=%u inputs=%u coefficients=%u shareds=%u entry=%u\n"
      "local_size=%u,%u,%u local_index=%u,%u workgroup_id=%u,%u num_workgroups=%u,%u\n"
      "ubo=%u,%u ssbo=%u,%u push=%u,%u ubo_used=0x%x ssbo_used=0x%x read=0x%x write=0x%x\n"
      "workgroup_shared=%u,%u,%u\n",
      &a.stage.temps, &a.stage.vertex_inputs, &a.stage.coefficients,
      &a.stage.shareds, &a.stage.entry_offset,
      &a.local_size[0], &a.local_size[1], &a.local_size[2],
      &a.local_invocation_index_start, &a.local_invocation_index_count,
      &a.workgroup_id_start, &a.workgroup_id_count,
      &a.num_workgroups_start, &a.num_workgroups_count,
      &a.stage.uniform_buffer_descriptor_start, &a.stage.uniform_buffer_descriptor_count,
      &a.storage_buffer_descriptor_start, &a.storage_buffer_descriptor_count,
      &a.stage.push_constant_start, &a.stage.push_constant_count,
      &a.uniform_buffer_used_mask, &a.storage_buffer_used_mask,
      &a.storage_buffer_read_mask, &a.storage_buffer_write_mask,
      &a.shared_memory_bytes, &a.shared_memory_descriptor_start,
      &a.shared_memory_descriptor_count);
  std::fclose(file);
  Check(count == 27, "incomplete compiled ABI");
  return a;
}

constexpr std::uint64_t kAddress = UINT64_C(0x120000fff0);
constexpr std::uint32_t kSentinel = UINT32_C(0xa55af00d);
struct Memory {
  std::vector<std::uint32_t> words, expected;
  std::vector<bool> writable, written;
  std::uint64_t reads = 0, stores = 0, stored_words = 0;
};
void Read(void *opaque, std::uint64_t, std::uint32_t, std::uint32_t *) {
  ++static_cast<Memory *>(opaque)->reads;
  throw std::runtime_error("write-only masked store invented a memory read");
}
void Write(void *opaque, std::uint64_t address, std::uint32_t count,
           const std::uint32_t *values) {
  auto &m = *static_cast<Memory *>(opaque);
  Check(address >= kAddress && (address - kAddress) % 4 == 0 && count != 0,
        "invalid store address or empty burst");
  const auto start = (address - kAddress) / 4;
  Check(start <= m.words.size() && count <= m.words.size() - start,
        "store exceeds the writable buffer");
  ++m.stores;
  for (unsigned i = 0; i < count; ++i) {
    const auto word = start + i;
    Check(m.writable[word], "native store touched a masked component, padding, or inactive lane");
    Check(!m.written[word], "native store wrote the same component more than once");
    Check(values[i] == m.expected[word], "native store used the wrong source component or offset");
    m.written[word] = true;
    m.words[word] = values[i];
    ++m.stored_words;
  }
}

void Test(const std::string &directory, unsigned kind, unsigned width,
          unsigned mask, bool branch) {
  const auto base = directory + "/compute-" + std::to_string(kind);
  auto abi = ReadAbi(base + ".abi");
  std::ifstream file(base + ".bin", std::ios::binary);
  Check(file.good(), "cannot open compiled PCO");
  const std::vector<std::uint8_t> binary{std::istreambuf_iterator<char>(file), {}};
  const auto program = DecodeComputePcoProgram(binary);
  ValidateComputeProgram(program, abi);
  Check(abi.local_size == std::array<std::uint32_t, 3>{37, 1, 1}, "lost workgroup shape");
  Check(abi.storage_buffer_read_mask == 0, "unexpected destination read permission");
  const unsigned binding = branch ? 3 : 0;
  Check(abi.storage_buffer_write_mask == (mask ? 1u << binding : 0), "incorrect selected buffer ABI");
  const unsigned stride = branch ? (width + 16) & ~15u : ((width + 3) & ~3u) + 4;
  const unsigned record_offset = branch ? 1 : 4;
  Memory memory;
  memory.words.assign(4 + 74 * stride + 4, kSentinel);
  memory.expected = memory.words;
  memory.writable.resize(memory.words.size());
  memory.written.resize(memory.words.size());
  std::uint64_t expected_stores = 0;
  for (unsigned index = 0; index < 74; ++index) {
    if (branch && (index % 37) % 2 == 0) continue;
    for (unsigned c = 0; c < width; ++c) {
      if (!(mask & (1u << c))) continue;
      const auto word = record_offset + index * stride + c;
      memory.writable[word] = true;
      memory.expected[word] = 0x12340000 + index * 256 + c;
      ++expected_stores;
    }
  }
  std::vector<std::uint32_t> shared(abi.stage.shareds);
  if (mask) {
    const auto descriptor = abi.storage_buffer_descriptor_start + binding * 4;
    shared.at(descriptor) = static_cast<std::uint32_t>(kAddress);
    shared.at(descriptor + 1) = static_cast<std::uint32_t>(kAddress >> 32);
    shared.at(descriptor + 2) = memory.words.size() * 4;
    shared.at(descriptor + 3) = 0;
  }
  ComputeMemoryCallbacks callbacks;
  callbacks.user_data = &memory;
  callbacks.read = Read;
  callbacks.write = Write;
  ComputeWorkgroupResult result;
  for (unsigned group = 0; group < 2; ++group) {
    for (unsigned first = 0; first < 37; first += kComputeTaskWidth) {
      const unsigned lanes = std::min(kComputeTaskWidth, 37 - first);
      auto task = MakeComputeTask(abi, shared, {2, 1, 1}, {group, 0, 0}, first, lanes);
      while (!task.ended) {
        Check(task.steps < 10000, "native masked store did not terminate");
        StepComputeTask(program, abi, task, callbacks, result);
      }
    }
  }
  Check(memory.reads == 0 && memory.stored_words == expected_stores,
        "native masked stores changed the allowed memory footprint");
  for (unsigned word = 0; word < memory.words.size(); ++word) {
    Check(memory.words[word] == memory.expected[word], "masked store damaged a guard or missed a value");
    Check(memory.written[word] == memory.writable[word], "selected store component did not execute exactly once");
  }
}
} // namespace

int main(int argc, char **argv) {
  try {
    Check(argc == 2, "usage: pco-compute-masked-store-test FIXTURE_DIRECTORY");
    std::ifstream manifest(std::string(argv[1]) + "/masked-stores.txt");
    Check(manifest.good(), "cannot open masked store manifest");
    unsigned kind, width, mask, branch, cases = 0;
    while (manifest >> kind >> width >> mask >> branch) {
      try { Test(argv[1], kind, width, mask, branch != 0); }
      catch (const std::exception &error) {
        throw std::runtime_error("fixture " + std::to_string(kind) + " width=" +
            std::to_string(width) + " mask=" + std::to_string(mask) +
            " branch=" + std::to_string(branch) + ": " + error.what());
      }
      ++cases;
    }
    Check(manifest.eof() && cases == 96, "incomplete masked store matrix");
    std::cout << "native compute masked stores: PASS (" << cases << " programs, "
              << checks << " checks; no destination reads or unselected writes)\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
