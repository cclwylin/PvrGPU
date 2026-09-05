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
  std::uint32_t width = 0;
  std::uint32_t height = 0;

  bool valid() const {
    return width != 0 && height != 0 &&
           static_cast<std::uint64_t>(pixels.size()) ==
               static_cast<std::uint64_t>(width) * height * 4U;
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
