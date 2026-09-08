// SPDX-License-Identifier: MIT
#include "common/centroid.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <random>

namespace {
unsigned checks = 0;
void Check(bool condition) {
  ++checks;
  if (!condition)
    throw std::runtime_error("centroid geometric contract failed");
}
void Test(std::uint32_t x, std::uint32_t y, float length, bool reverse) {
  float tx[3]{float(x), float(x) + length, float(x)};
  float ty[3]{float(y), float(y), float(y) + length};
  if (reverse) {
    std::swap(tx[1], tx[2]);
    std::swap(ty[1], ty[2]);
  }
  const std::array<float, 2> preferred{float(x) + .75f, float(y) + .75f};
  const auto result = pvrgpu::stub::CentroidInsidePrimitive(tx, ty, x, y, preferred);
  Check(result[0] >= x && result[0] <= x + 1 && result[1] >= y && result[1] <= y + 1);
  const double dx = result[0] - double(x), dy = result[1] - double(y);
  Check(dx >= 0 && dy >= 0 && dx + dy <= double(tx[reverse ? 2 : 1]) - x);
  if (length > 1.5f)
    Check(result == preferred);
  else
    Check(result != preferred);
}
}

int main() {
  try {
    for (unsigned x : {0U, 5U, 127U, 1023U})
      for (unsigned y : {0U, 29U, 255U, 1023U})
        for (float length : {0.125f, .5f, 1.f, 1.4999f, 1.5f, 1.5002f, 2.f, 4.f})
          for (bool reverse : {false, true})
            Test(x, y, length, reverse);
    std::mt19937 random(0x63656e74);
    std::uniform_real_distribution<float> extent(.125f, 4.f);
    for (unsigned i = 0; i < 20000; ++i)
      Test(random() % 1024, random() % 1024, extent(random), random() & 1);
    {
      // A real 16x dEQP narrow-fan edge: fixed-point coverage selected the
      // preferred sample at (73, 88.5), but evaluating float interpolation
      // planes there can round one barycentric just below zero.  The centroid
      // must move to a stable interior point.
      float edge_x[3]{76.8f, 69.5652f, 67.2886f};
      float edge_y[3]{44.8f, 128.0f, 128.0f};
      const std::array<float, 2> preferred{73.0f, 88.5f};
      const auto result = pvrgpu::stub::CentroidInsidePrimitive(
          edge_x, edge_y, 73, 88, preferred);
      Check(result != preferred);
      const auto cross = [](float ax, float ay, float bx, float by,
                            float px, float py) {
        return double(bx - ax) * double(py - ay) -
               double(by - ay) * double(px - ax);
      };
      const double area = cross(edge_x[0], edge_y[0], edge_x[1], edge_y[1],
                                edge_x[2], edge_y[2]);
      const double sign = area > 0 ? 1.0 : -1.0;
      for (unsigned edge = 0; edge < 3; ++edge) {
        const unsigned next = (edge + 1) % 3;
        Check(sign * cross(edge_x[edge], edge_y[edge], edge_x[next],
                           edge_y[next], result[0], result[1]) > 0.0);
      }
    }
    float tx[3]{0, 1, 0}, ty[3]{0, 0, 1};
    bool rejected = false;
    try { pvrgpu::stub::CentroidInsidePrimitive(tx, ty, 0, 0, {2, .5f}); }
    catch (const std::runtime_error &) { rejected = true; }
    Check(rejected);
    tx[0] = std::numeric_limits<float>::quiet_NaN();
    rejected = false;
    try { pvrgpu::stub::CentroidInsidePrimitive(tx, ty, 0, 0, {.5f, .5f}); }
    catch (const std::runtime_error &) { rejected = true; }
    Check(rejected);
    std::cout << "centroid_geometry_test: PASS " << checks << " checks\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << " after " << checks << " checks\n";
    return 1;
  }
}
