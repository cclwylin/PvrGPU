#pragma once

#include "model_types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace pvrgpu::stub {

bool ConfigureDriverCommandOptions(Options *options, std::string *error);

/*
 * The pixels a flush left in DRAM, as RGBA8.
 *
 * This is the model's own framebuffer readback, not a command sidecar: it is
 * what a `glReadPixels` on a colour attachment has to return, and until it
 * came back the driver could only ever hand out its own CPU clear.
 */
struct ModelFramebuffer {
  std::vector<std::uint8_t> pixels;
  // Colour attachments past the first, in target order.  Each is the same
  // width, height and pixel width as `pixels`.
  std::vector<std::vector<std::uint8_t>> extra;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  // Four while the attachment packs UNORM8 channels; an integer attachment
  // stores one 32-bit channel per dword and is 8 or 16 bytes wide.
  std::uint32_t bytes_per_pixel = 4;
  // Samples are stored next to each other within each pixel, without resolve.
  std::uint32_t sample_count = 1;
  std::vector<std::uint8_t> depth_pixels;
  std::uint32_t depth_format = 0;

  bool valid() const {
    return width != 0 && height != 0 && bytes_per_pixel != 0 &&
           sample_count != 0 &&
           static_cast<std::uint64_t>(pixels.size()) ==
               static_cast<std::uint64_t>(width) * height * bytes_per_pixel *
                   sample_count;
  }
};

/*
 * Run one flush on the persistent model.
 *
 * The model elaborates on the first call and stays alive afterwards, so this
 * may be called once per readback rather than once per process.
 * `framebuffer`, when given, receives the flush's final DRAM readback.
 */
int RunConfiguredModel(Options options,
                       ModelFramebuffer *framebuffer = nullptr);

/*
 * End the simulation.  `sc_stop()` is one-way: after it no further flush will
 * run, so this belongs at teardown and nowhere else.
 */
void ShutdownConfiguredModel();

}  // namespace pvrgpu::stub
