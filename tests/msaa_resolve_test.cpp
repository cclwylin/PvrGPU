/* SPDX-License-Identifier: MIT */
#include "gallium/drivers/pvrgpu/pvrgpu_msaa.h"

#include <array>
#include <iostream>
#include <stdexcept>

namespace {
void Require(bool condition, const char *message) {
   if (!condition)
      throw std::runtime_error(message);
}

void ResolveAllSamples() {
   struct Sample {
      float color[4];
      float padding[3];
   };
   std::array<Sample, 16> samples{};
   for (unsigned sample = 0; sample < samples.size(); ++sample) {
      samples[sample].color[0] = static_cast<float>(sample);
      samples[sample].color[1] = sample == 0 ? 1.0f : 0.0f;
      samples[sample].color[2] = -static_cast<float>(sample);
      samples[sample].color[3] = 1234.5f;
      for (float &padding : samples[sample].padding)
         padding = -100000.0f;
   }
   for (unsigned count : {1u, 2u, 4u, 8u, 16u}) {
      float resolved[4]{};
      pvrgpu_msaa_resolve_float(samples.data(), count, sizeof(Sample), resolved);
      Require(resolved[0] == (count - 1) * 0.5f,
              "resolve did not average distinct sample colors");
      Require(resolved[1] == 1.0f / count,
              "resolve replicated sample zero instead of averaging coverage");
      Require(resolved[2] == -static_cast<float>(count - 1) * 0.5f,
              "resolve clamped negative floating-point colors");
      Require(resolved[3] == 1234.5f,
              "resolve clamped HDR floating-point colors");
   }
}

void InterleavedAddressing() {
   for (unsigned count : {1u, 2u, 4u, 8u, 16u}) {
      for (unsigned pixel = 0; pixel < 17; ++pixel) {
         for (unsigned sample = 0; sample < count; ++sample) {
            Require(pvrgpu_msaa_texel_index(pixel, sample, count) ==
                       static_cast<size_t>(pixel) * count + sample,
                    "sample address crossed a pixel boundary");
         }
      }
   }
}
} // namespace

int main() {
   try {
      ResolveAllSamples();
      InterleavedAddressing();
      std::cout << "MSAA resolve tests passed\n";
      return 0;
   } catch (const std::exception &error) {
      std::cerr << error.what() << '\n';
      return 1;
   }
}
