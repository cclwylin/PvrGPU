// SPDX-License-Identifier: MIT
#ifndef PVRGPU_SYSTEMC_COMMON_CENTROID_H
#define PVRGPU_SYSTEMC_COMMON_CENTROID_H

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace pvrgpu::stub {

// Coverage uses snapped fixed-point edges, while interpolation planes use
// the original floating-point vertices. An edge sample can therefore pass
// coverage and still lie outside the interpolation primitive. Keep the
// standard sample/center choice when it is inside; otherwise choose the
// mean of the triangle/pixel intersection's vertices. This is fixed-function
// geometry only: no varying, coefficient, or shader result is inspected.
// GLSL ES 3.20 section 4.5 requires centroid locations inside both regions.
inline std::array<float, 2> CentroidInsidePrimitive(
    const float (&triangle_x)[3], const float (&triangle_y)[3],
    std::uint32_t pixel_x, std::uint32_t pixel_y,
    const std::array<float, 2> &preferred) {
  using Point = std::array<double, 2>;
  std::array<Point, 8> polygon{};
  const double left = pixel_x, bottom = pixel_y;
  for (unsigned i = 0; i < 3; ++i) {
    if (!std::isfinite(triangle_x[i]) || !std::isfinite(triangle_y[i]))
      throw std::runtime_error("centroid primitive contains non-finite geometry");
    polygon[i] = {triangle_x[i], triangle_y[i]};
  }
  if (!std::isfinite(preferred[0]) || !std::isfinite(preferred[1]) ||
      preferred[0] < left || preferred[0] > left + 1 ||
      preferred[1] < bottom || preferred[1] > bottom + 1)
    throw std::runtime_error("centroid preferred location is outside its pixel");
  const auto cross = [](const Point &a, const Point &b, const Point &p) {
    return (b[0] - a[0]) * (p[1] - a[1]) -
           (b[1] - a[1]) * (p[0] - a[0]);
  };
  const double area = cross(polygon[0], polygon[1], polygon[2]);
  if (area == 0)
    return preferred; // Only the snapped raster primitive has nonzero area.
  const double sign = area > 0 ? 1 : -1;
  // A fixed-point coverage sample can map back to a point only a few float
  // ulps inside an unsnapped edge.  The interpolation plane is float as well,
  // so treating that point as comfortably interior can still produce a tiny
  // negative barycentric value.  Require a scale-relative geometric margin;
  // otherwise use the clipped polygon's interior mean below.
  const double interior_margin =
      std::max(1.0e-12, std::abs(area) * 1.0e-6);
  const Point chosen{preferred[0], preferred[1]};
  bool inside = true;
  for (unsigned edge = 0; edge < 3; ++edge)
    inside &= sign * cross(polygon[edge], polygon[(edge + 1) % 3], chosen) >
              interior_margin;
  if (inside)
    return preferred;

  unsigned count = 3;
  for (unsigned plane = 0; plane < 4 && count; ++plane) {
    const unsigned axis = plane / 2;
    const double boundary = (axis == 0 ? left : bottom) + (plane & 1);
    const auto distance = [&](const Point &point) {
      return (plane & 1) ? boundary - point[axis] : point[axis] - boundary;
    };
    std::array<Point, 8> clipped{};
    unsigned next_count = 0;
    auto append = [&](const Point &point) {
      if (next_count == clipped.size())
        throw std::runtime_error("centroid clipped polygon exceeds its geometric bound");
      clipped[next_count++] = point;
    };
    Point previous = polygon[count - 1];
    double previous_distance = distance(previous);
    for (unsigned index = 0; index < count; ++index) {
      const Point current = polygon[index];
      const double current_distance = distance(current);
      if ((previous_distance >= 0) != (current_distance >= 0)) {
        const double t = previous_distance / (previous_distance - current_distance);
        Point crossing{previous[0] + t * (current[0] - previous[0]),
                       previous[1] + t * (current[1] - previous[1])};
        crossing[axis] = boundary;
        append(crossing);
      }
      if (current_distance >= 0)
        append(current);
      previous = current;
      previous_distance = current_distance;
    }
    polygon = clipped;
    count = next_count;
  }
  if (count < 3)
    return preferred; // No two-dimensional original-geometry intersection.
  Point center{};
  for (unsigned i = 0; i < count; ++i) {
    center[0] += polygon[i][0] - left;
    center[1] += polygon[i][1] - bottom;
  }
  return {static_cast<float>(left + center[0] / count),
          static_cast<float>(bottom + center[1] / count)};
}

} // namespace pvrgpu::stub
#endif
