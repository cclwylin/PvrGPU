/* SPDX-License-Identifier: MIT */
#include "gallium/drivers/pvrgpu/pvrgpu_msaa.h"

#include <array>
#include <cfenv>
#include <cmath>
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

void ResolveArithmeticOrder() {
   // This checks the explicit Mesa shader operation order under ordinary
   // IEEE round-to-nearest semantics, without fast-math/reassociation. It is
   // not a requirement on an LLVM build that opts out of signed-zero rules.
   Require(std::fegetround() == FE_TONEAREST,
           "resolve arithmetic test requires round-to-nearest");
   std::array<std::array<float, 4>, 16> samples{};
   for (unsigned sample = 0; sample < samples.size(); ++sample) {
      samples[sample] = {-0.0f, +0.0f, sample == 0 ? 1.0f : -0.0f,
                         sample % 2 == 0 ? -0.0f : +0.0f};
   }
   for (unsigned count : {1u, 2u, 4u, 8u, 16u}) {
      float resolved[4]{};
      pvrgpu_msaa_resolve_float(samples.data(), count, sizeof(samples[0]),
                                resolved);
      for (unsigned channel : {0u, 1u, 3u}) {
         Require(resolved[channel] == 0.0f && !std::signbit(resolved[channel]),
                 "resolve must accumulate samples starting from positive zero");
      }
      Require(resolved[2] == 1.0f / count,
              "signed-zero samples changed a nonzero sample contribution");
   }

   // Sequential F32 addition loses the second sample's unit before the
   // third sample cancels 2^24. Reassociation would change the result.
   const float ordered[4][4] = {{16777216.0f, 0.0f, 0.0f, 0.0f},
                                 {1.0f, 0.0f, 0.0f, 0.0f},
                                 {-16777216.0f, 0.0f, 0.0f, 0.0f},
                                 {1.0f, 0.0f, 0.0f, 0.0f}};
   float resolved[4]{};
   pvrgpu_msaa_resolve_float(ordered, 4, sizeof(ordered[0]), resolved);
   Require(resolved[0] == 0.25f,
           "resolve did not retain ascending-sample F32 addition order");
}
} // namespace

int main() {
   try {
      ResolveAllSamples();
      InterleavedAddressing();
      ResolveArithmeticOrder();
      std::cout << "MSAA resolve tests passed\n";
      return 0;
   } catch (const std::exception &error) {
      std::cerr << error.what() << '\n';
      return 1;
   }
}
