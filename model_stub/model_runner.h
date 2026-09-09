#pragma once

#include "model_types.h"
#include "graphics_stats.h"

#include <cstdint>
#include <string>
#include <vector>

namespace pvrgpu::stub {

struct ModelComputeDispatch;
struct ModelComputeStats;

bool ConfigureDriverCommandOptions(Options *options, std::string *error);

/*
 * The pixels a flush left in DRAM, as RGBA8.
 *
 * This is the model's own framebuffer readback, not a command sidecar: it is
 * what a `glReadPixels` on a colour attachment has to return, and until it
 * came back the driver could only ever hand out its own CPU clear.
 */
struct ModelFramebuffer {
  ModelGraphicsStats graphics_stats;
  std::vector<ModelStreamOutputReadback> stream_outputs;
  std::vector<ModelShaderImageReadback> shader_images;
  std::vector<std::uint8_t> pixels;
  // Colour attachments past the first, in target order.  Each is the same
  // width, height and pixel width as `pixels`.
  std::vector<std::vector<std::uint8_t>> extra;
  // Actual transport identities in target order; bytes-per-pixel alone
  // cannot distinguish RGBA8 from either packed 10/10/10/2 layout.
  std::vector<std::string> color_formats;
  bool color_formats_explicit = false;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  // Four while the attachment packs UNORM8 channels; an integer attachment
  // stores one 32-bit channel per dword and is 8 or 16 bytes wide.
  std::uint32_t bytes_per_pixel = 4;
  // Samples are stored next to each other within each pixel, without resolve.
  std::uint32_t sample_count = 1;
  std::uint32_t layer_count = 1;
  std::vector<std::uint8_t> depth_pixels;
  std::uint32_t depth_format = 0;

  bool valid() const {
    return width != 0 && height != 0 && width <= 4096 && height <= 4096 &&
           bytes_per_pixel != 0 && bytes_per_pixel <= 16 &&
           sample_count != 0 && sample_count <= 16 &&
           layer_count != 0 && layer_count <= 256 &&
           static_cast<std::uint64_t>(pixels.size()) ==
               static_cast<std::uint64_t>(width) * height * bytes_per_pixel *
                   sample_count * layer_count;
  }

  bool ColorFormatMatches(std::uint32_t target, const char *requested) const {
    if (target > extra.size() ||
        (!color_formats.empty() && color_formats.size() != extra.size() + 1U) ||
        (color_formats_explicit && color_formats.size() != extra.size() + 1U))
      return false;
    if (color_formats_explicit) {
      if (color_formats.size() > 4U || bytes_per_pixel != 4U)
        return false;
      for (const auto &format : color_formats)
        if (!IsNormalizedFourByteColorFormat(format))
          return false;
    }
    if (!requested)
      return !color_formats_explicit;
    return requested[0] && target < color_formats.size() &&
           color_formats[target] == requested;
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

// Compute uses the same elaborated memory/session, with its own CDM and
// ComputeShader modules. Writable raw resources are published on success.
int RunConfiguredCompute(ModelComputeDispatch *dispatch,
                          ModelComputeStats *stats, std::string *error);

/*
 * End the simulation.  `sc_stop()` is one-way: after it no further flush will
 * run, so this belongs at teardown and nowhere else.
 */
void ShutdownConfiguredModel();

}  // namespace pvrgpu::stub
