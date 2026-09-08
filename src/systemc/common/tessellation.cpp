// SPDX-License-Identifier: MIT
// Domain/mode mapping follows Mesa p_tessellator.cpp and draw_tess.c.
// Copyright 2020 Red Hat. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "common/tessellation.h"

#include "common/tessellation_reference.h"

#include <limits>
#include <stdexcept>

namespace pvrgpu::stub {

TessellationStatus TessellatePatch(const TessellationRequest &request,
                                 TessellationStorage storage,
                                 TessellationResult &result) {
  using namespace tessellation_reference;
  result = {};
  if (request.domain != TessellationDomain::kTriangles &&
      request.domain != TessellationDomain::kQuads &&
      request.domain != TessellationDomain::kIsolines)
    return TessellationStatus::kInvalidDomain;
  PIPE_TESSELLATOR_PARTITIONING partitioning;
  switch (request.spacing) {
    case TessellationSpacing::kEqual:
      partitioning = PIPE_TESSELLATOR_PARTITIONING_INTEGER;
      break;
    case TessellationSpacing::kFractionalEven:
      partitioning = PIPE_TESSELLATOR_PARTITIONING_FRACTIONAL_EVEN;
      break;
    case TessellationSpacing::kFractionalOdd:
      partitioning = PIPE_TESSELLATOR_PARTITIONING_FRACTIONAL_ODD;
      break;
    default: return TessellationStatus::kInvalidSpacing;
  }
  if (request.clockwise > 1 || request.point_mode > 1)
    return TessellationStatus::kInvalidFlags;
  if (!storage.points || !storage.indices)
    return TessellationStatus::kInvalidStorage;
  if (storage.point_capacity < kTessellationMaxPoints ||
      storage.index_capacity < kTessellationMaxIndices)
    return TessellationStatus::kInsufficientCapacity;
  const auto point_address = reinterpret_cast<std::uintptr_t>(storage.points);
  const auto index_address = reinterpret_cast<std::uintptr_t>(storage.indices);
  constexpr auto point_bytes = kTessellationMaxPoints * sizeof(TessellationDomainPoint);
  constexpr auto index_bytes = kTessellationMaxIndices * sizeof(std::uint32_t);
  if (point_address % alignof(TessellationDomainPoint) ||
      index_address % alignof(std::uint32_t) ||
      point_address > std::numeric_limits<std::uintptr_t>::max() - point_bytes ||
      index_address > std::numeric_limits<std::uintptr_t>::max() - index_bytes ||
      (point_address < index_address + index_bytes &&
       index_address < point_address + point_bytes))
    return TessellationStatus::kInvalidStorage;
  PIPE_TESSELLATOR_OUTPUT_PRIMITIVE output;
  if (request.point_mode) {
    output = PIPE_TESSELLATOR_OUTPUT_POINT;
    result.primitive_size = 1;
  } else if (request.domain == TessellationDomain::kIsolines) {
    output = PIPE_TESSELLATOR_OUTPUT_LINE;
    result.primitive_size = 2;
  } else {
    // draw_tess.c passes !shader->vertex_order_cw into p_tess_init: the
    // D3D reference and GL domain orientation are opposite conventions.
    output = request.clockwise ? PIPE_TESSELLATOR_OUTPUT_TRIANGLE_CCW
                               : PIPE_TESSELLATOR_OUTPUT_TRIANGLE_CW;
    result.primitive_size = 3;
  }
  CHWTessellator tessellator(storage.points, storage.indices);
  tessellator.Init(partitioning, output);
  try {
    switch (request.domain) {
      case TessellationDomain::kTriangles:
        tessellator.TessellateTriDomain(request.outer[0], request.outer[1],
                                      request.outer[2], request.inner[0]);
        break;
      case TessellationDomain::kQuads:
        tessellator.TessellateQuadDomain(request.outer[0], request.outer[1],
                                       request.outer[2], request.outer[3],
                                       request.inner[0], request.inner[1]);
        break;
      case TessellationDomain::kIsolines:
        tessellator.TessellateIsoLineDomain(request.outer[0], request.outer[1]);
        break;
    }
  } catch (const std::out_of_range &) {
    result = {};
    return TessellationStatus::kInternalRange;
  }
  const int points = tessellator.GetPointCount();
  const int indices = tessellator.GetIndexCount();
  if (points < 0 || points > static_cast<int>(kTessellationMaxPoints) ||
      indices < 0 || indices > static_cast<int>(kTessellationMaxIndices) ||
      indices % result.primitive_size != 0) {
    result = {};
    return TessellationStatus::kInternalRange;
  }
  for (int index = 0; index < indices; ++index) {
    if (storage.indices[index] >= static_cast<unsigned>(points)) {
      result = {};
      return TessellationStatus::kInternalRange;
    }
  }
  result.point_count = static_cast<std::uint32_t>(points);
  result.index_count = static_cast<std::uint32_t>(indices);
  return TessellationStatus::kSuccess;
}

float TessellationCoordinateW(TessellationDomain domain,
                             TessellationDomainPoint point) {
  return domain == TessellationDomain::kTriangles ? (1.0f - point.u) - point.v
                                                 : 0.0f;
}

const char *TessellationStatusName(TessellationStatus status) {
  switch (status) {
    case TessellationStatus::kSuccess: return "success";
    case TessellationStatus::kInvalidDomain: return "invalid tessellation domain";
    case TessellationStatus::kInvalidSpacing: return "invalid tessellation spacing";
    case TessellationStatus::kInvalidFlags: return "invalid tessellation flags";
    case TessellationStatus::kInvalidStorage: return "invalid tessellation storage";
    case TessellationStatus::kInsufficientCapacity: return "insufficient tessellation capacity";
    case TessellationStatus::kInternalRange: return "tessellation internal range";
  }
  return "unknown tessellation status";
}

}  // namespace pvrgpu::stub
