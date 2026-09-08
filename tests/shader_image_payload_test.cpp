#include "shader_images.h"
#include "uniform_buffers.h"
#include <iostream>
#include <stdexcept>

using namespace pvrgpu::stub;
namespace {
unsigned checks = 0;
void Check(bool value) {
  ++checks;
  if (!value) throw std::runtime_error("fragment image payload contract failed");
}
DriverCommand Fixture() {
  DriverCommand command;
  command.fragment_image_descriptor_count = 1;
  command.fragment_image_read_mask = command.fragment_image_write_mask = 1;
  command.fragment_shared = {0, 0, 1, 4, 1, 1, 4, 4};
  command.fragment_pco_abi.shareds = command.fragment_pco_abi.push_constant_start = 8;
  DriverShaderImage image;
  image.format = 1; image.access = 3; image.resource_token = 7;
  image.bytes = {0xa5, 0xa5, 0xa5, 0xa5, 0, 0, 0, 0, 0x5a, 0x5a, 0x5a, 0x5a};
  image.offset = 4; image.width = image.height = image.depth = 1;
  image.row_stride = image.layer_stride = image.texel_bytes = 4;
  command.fragment_images.push_back(image);
  return command;
}
template<class F> void Reject(F mutate) {
  auto command = Fixture(); mutate(command);
  std::string error;
  Check(!ValidateDriverShaderImages(command, &error) && !error.empty());
}
}
int main() {
  try {
    std::string error;
    Check(ValidateDriverShaderImages(DriverCommand{}, &error));
    auto command = Fixture();
    Check(ValidateDriverShaderImages(command, &error));
    Check(ValidateDriverUniformBuffers(command, &error));
    Reject([](auto &c){ c.fragment_image_descriptor_count = 33; });
    Reject([](auto &c){ c.fragment_early_tests = 2; });
    Reject([](auto &c){ c.fragment_image_write_mask = 2; });
    Reject([](auto &c){ c.fragment_image_descriptor_start = 1; });
    Reject([](auto &c){ c.fragment_shared[0] = 1; });
    Reject([](auto &c){ c.fragment_shared[7] = 8; });
    Reject([](auto &c){ c.fragment_shared.pop_back(); });
    Reject([](auto &c){ c.fragment_images.clear(); });
    Reject([](auto &c){ c.fragment_images.push_back(c.fragment_images[0]); });
    Reject([](auto &c){ c.fragment_images[0].image_slot = 1; });
    Reject([](auto &c){ c.fragment_images[0].format = 2; });
    Reject([](auto &c){ c.fragment_images[0].access = 1; });
    Reject([](auto &c){ c.fragment_images[0].resource_token = 0; });
    Reject([](auto &c){ c.fragment_images[0].offset = UINT64_MAX; });
    Reject([](auto &c){ c.fragment_images[0].offset = 5; });
    Reject([](auto &c){ c.fragment_images[0].offset = 12; });
    Reject([](auto &c){ c.fragment_images[0].width = 2; });
    Reject([](auto &c){ c.fragment_images[0].height = UINT32_MAX; });
    Reject([](auto &c){ c.fragment_images[0].depth = UINT32_MAX; });
    Reject([](auto &c){ c.fragment_images[0].bytes.clear(); });
    command.fragment_image_descriptor_count = 2;
    command.fragment_image_read_mask = command.fragment_image_write_mask = 3;
    command.fragment_pco_abi.shareds = command.fragment_pco_abi.push_constant_start = 16;
    command.fragment_shared.insert(command.fragment_shared.end(), {0,0,1,4,1,1,4,4});
    command.fragment_images.push_back(command.fragment_images[0]);
    command.fragment_images[1].image_slot = 1;
    command.fragment_images[1].offset = 8;
    Check(ValidateDriverShaderImages(command, &error));
    command.fragment_images[1].bytes[0] ^= 1;
    Check(!ValidateDriverShaderImages(command, &error));
    command = Fixture();
    command.fragment_image_descriptor_count = 32;
    command.fragment_image_read_mask = command.fragment_image_write_mask = UINT32_C(1) << 31;
    command.fragment_images[0].image_slot = 31;
    command.fragment_shared.assign(256, 0);
    const auto descriptor = Fixture().fragment_shared;
    std::copy(descriptor.begin(), descriptor.end(), command.fragment_shared.begin() + 248);
    command.fragment_pco_abi.shareds = command.fragment_pco_abi.push_constant_start = 256;
    Check(ValidateDriverShaderImages(command, &error));
    command.fragment_shared[8] = 1;
    Check(!ValidateDriverShaderImages(command, &error));
    command = Fixture();
    command.fragment_sampled_texture_count = 1;
    command.fragment_image_descriptor_start = 24;
    command.fragment_pco_abi.uniform_buffer_descriptor_start = 20;
    command.fragment_pco_abi.uniform_buffer_descriptor_count = 1;
    command.fragment_pco_abi.push_constant_start = 32;
    command.fragment_pco_abi.push_constant_count = 4;
    command.fragment_pco_abi.shareds = 36;
    command.fragment_shared.assign(36, 0);
    command.fragment_shared[22] = 16;
    std::copy(descriptor.begin(), descriptor.end(), command.fragment_shared.begin() + 24);
    DriverPcoUniformBuffer ubo;
    ubo.stage = DriverPcoShaderStage::kFragment;
    ubo.bytes.resize(16, 0x3c);
    command.uniform_buffers.push_back(ubo);
    Check(ValidateDriverUniformBuffers(command, &error));
    Check(ValidateDriverShaderImages(command, &error));
    std::cout << "shader_image_payload_test: PASS " << checks << " checks\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << " after " << checks << " checks\n";
    return 1;
  }
}
