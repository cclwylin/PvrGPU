#include "../model_stub/uniform_buffers.h"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace pvrgpu::stub;

namespace {
[[noreturn]] void Fail(const char *message) {
  std::fprintf(stderr, "uniform-buffer-contract-test: %s\n", message);
  std::exit(EXIT_FAILURE);
}

DriverCommand Fixture() {
  DriverCommand command;
  command.vertex_pco_abi.shareds = 9;
  command.vertex_pco_abi.uniform_buffer_descriptor_count = 2;
  command.vertex_pco_abi.push_constant_start = 8;
  command.vertex_pco_abi.push_constant_count = 1;
  command.vertex_shared = {0, 0, 0, 0, 0, 0, 5, 0, 0x11223344};
  command.fragment_sampled_texture_count = 2;
  command.fragment_pco_abi.shareds = 46;
  command.fragment_pco_abi.uniform_buffer_descriptor_start = 40;
  command.fragment_pco_abi.uniform_buffer_descriptor_count = 1;
  command.fragment_pco_abi.push_constant_start = 44;
  command.fragment_pco_abi.push_constant_count = 2;
  command.fragment_shared.assign(46, 0);
  command.fragment_shared[42] = 7;
  command.uniform_buffers = {
      {DriverPcoShaderStage::kVertex, 1, {1, 2, 3, 4, 5}},
      {DriverPcoShaderStage::kFragment, 0, {9, 8, 7, 6, 5, 4, 3}},
  };
  return command;
}

void Accept(const DriverCommand &command) {
  std::string error;
  if (!ValidateDriverUniformBuffers(command, &error))
    Fail(error.c_str());
}

template <typename Mutation>
void Reject(Mutation mutation, const char *reason) {
  auto command = Fixture();
  mutation(command);
  std::string error;
  if (ValidateDriverUniformBuffers(command, &error) ||
      error.find(reason) == std::string::npos)
    Fail(reason);
}
}  // namespace

int main() {
  Accept(DriverCommand{});
  Accept(Fixture());  // Sparse VS index, independent FS index, two textures.
  auto maximum = Fixture();
  maximum.uniform_buffers[0].bytes.resize(kMaximumUniformBufferBytes);
  maximum.vertex_shared[6] = kMaximumUniformBufferBytes;
  Accept(maximum);
  Reject([](auto &c) { c.uniform_buffers.push_back(c.uniform_buffers[0]); },
         "duplicated");
  Reject([](auto &c) { c.uniform_buffers[0].stage = static_cast<DriverPcoShaderStage>(2); },
         "stage/index/size");
  Reject([](auto &c) { c.uniform_buffers[0].block_index = 15; }, "stage/index/size");
  Reject([](auto &c) { c.uniform_buffers[0].block_index = 2; }, "descriptor");
  Reject([](auto &c) { c.uniform_buffers[0].bytes.clear(); }, "stage/index/size");
  Reject([](auto &c) { c.uniform_buffers[0].bytes.resize(kMaximumUniformBufferBytes + 1); },
         "stage/index/size");
  Reject([](auto &c) { c.vertex_shared[4] = 128; }, "not canonical");
  Reject([](auto &c) { c.vertex_shared[5] = 1; }, "not canonical");
  Reject([](auto &c) { c.vertex_shared[6] = 4; }, "not canonical");
  Reject([](auto &c) { c.vertex_shared[7] = 4; }, "not canonical");
  Reject([](auto &c) { c.vertex_shared[2] = 4; }, "not canonical");
  Reject([](auto &c) { c.vertex_shared.pop_back(); }, "layout");
  Reject([](auto &c) { c.fragment_pco_abi.uniform_buffer_descriptor_start = 20; }, "layout");
  Reject([](auto &c) { c.fragment_pco_abi.push_constant_start = 40; }, "layout");
  Reject([](auto &c) { c.fragment_pco_abi.uniform_buffer_descriptor_count = UINT32_MAX; },
         "range");
  Reject([](auto &c) { c.fragment_pco_abi.uniform_buffer_descriptor_start = UINT32_MAX; },
         "layout");
  Reject([](auto &c) { c.fragment_pco_abi.push_constant_count = UINT32_MAX; }, "layout");
  const auto old_draw = Fixture();
  auto new_draw = old_draw;
  new_draw.uniform_buffers[0].bytes[0] = 77;
  if (old_draw.uniform_buffers[0].bytes[0] != 1)
    Fail("owned per-draw payload aliases another draw");
  std::puts("uniform-buffer-contract-test: PASS");
  return EXIT_SUCCESS;
}
