#include "common/pipeline_state.h"

#include <iostream>
#include <stdexcept>

int main() {
  using namespace pvrgpu::stub;
  unsigned checks = 0;
  const auto check = [&](bool value) {
    ++checks;
    if (!value) throw std::runtime_error("fragment output mask contract");
  };
  // All combinations of explicit code ownership, including ordinary FS
  // without GS/TES. Handle identity is tested, not a shader-name heuristic.
  for (unsigned code = 0; code < 8; ++code) {
    PipelineState state;
    state.fragment_code.generation = (code & 1) ? 1 : 0;
    state.geometry_code.generation = (code & 2) ? 2 : 0;
    state.tessellation_state.generation = (code & 4) ? 3 : 0;
    check(HasExplicitFragmentOutputMasks(state) == (code != 0));
    check(ExpectedPixelOutputMask(state.fragment_output_mask,
                                  HasExplicitFragmentOutputMasks(state)) ==
          (code ? 0U : 15U));
    // Color write-enable and shader depth flags cannot erase real exports
    // or invent exports for a shader that writes only depth.
    state.raster_state.shader_writes_depth = 1;
    check(ExpectedPixelOutputMask(state.fragment_output_mask,
                                  HasExplicitFragmentOutputMasks(state)) ==
          (code ? 0U : 15U));
    for (unsigned target = 0; target < 4; ++target) {
      for (unsigned mask = 1; mask <= 15; ++mask) {
        state.fragment_output_mask = {};
        state.fragment_output_mask[target] = mask;
        check(ExpectedPixelOutputMask(state.fragment_output_mask,
                                      HasExplicitFragmentOutputMasks(state)) ==
              (mask << (4 * target)));
      }
    }
    state.fragment_output_mask = {1, 3, 7, 15};
    check(ExpectedPixelOutputMask(state.fragment_output_mask,
                                  HasExplicitFragmentOutputMasks(state)) == 0xf731);
  }
  std::cout << "fragment output mask: " << checks << " checks PASS\n";
}
