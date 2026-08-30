// Capture-only TestBase implementation for official ChromeOS GLBench cases.

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstring>
#include <memory>
#include <string>

#include "filepath.h"
#include "glinterface.h"
#include "png_helper.h"
#include "testbase.h"
#include "utils.h"

namespace glbench {
namespace {

int g_captured_samples = 0;

typedef void (*RenderDocSetCaptureFilePathTemplate)(const char* path_template);
typedef void (*RenderDocStartFrameCapture)(void* device, void* window_handle);
typedef uint32_t (*RenderDocEndFrameCapture)(void* device,
                                             void* window_handle);
typedef int (*RenderDocGetAPI)(int version, void** api);

struct MinimalRenderDocAPI {
  void* unused_before_set_path[11];
  RenderDocSetCaptureFilePathTemplate SetCaptureFilePathTemplate;
  void* unused_before_start_capture[7];
  RenderDocStartFrameCapture StartFrameCapture;
  void* IsFrameCapturing;
  RenderDocEndFrameCapture EndFrameCapture;
};

MinimalRenderDocAPI* GetRenderDocAPI() {
  static MinimalRenderDocAPI* api = nullptr;
  static bool attempted = false;
  if (attempted)
    return api;
  attempted = true;
  RenderDocGetAPI get_api = reinterpret_cast<RenderDocGetAPI>(
      dlsym(RTLD_DEFAULT, "RENDERDOC_GetAPI"));
  if (get_api && !get_api(10700, reinterpret_cast<void**>(&api)))
    api = nullptr;
  return api;
}

bool StartRenderDocCapture() {
  const char* capture_path = getenv("RENDERDOC_CAPTURE_FILE");
  if (!capture_path || !capture_path[0])
    return false;
  MinimalRenderDocAPI* api = GetRenderDocAPI();
  if (!api) {
    std::fprintf(stderr,
                 "RenderDoc capture requested, but RENDERDOC_GetAPI is "
                 "unavailable.\n");
    return false;
  }
  api->SetCaptureFilePathTemplate(capture_path);
  api->StartFrameCapture(nullptr, nullptr);
  return true;
}

void EndRenderDocCapture(bool started) {
  if (!started)
    return;
  glFinish();
  MinimalRenderDocAPI* api = GetRenderDocAPI();
  if (!api || !api->EndFrameCapture(nullptr, nullptr))
    std::fprintf(stderr, "RenderDoc failed to finish GLBench capture.\n");
}

void SaveSampleImage(const char* testname, int sample, int width, int height) {
  const char* output = getenv("GLBENCH_CAPTURE_OUTPUT_DIR");
  if (!output || !output[0])
    return;
  FilePath directory(output);
  if (!CreateDirectory(directory)) {
    std::fprintf(stderr, "Could not create GLBench PNG directory: %s\n",
                 output);
    return;
  }

  char basename[512];
  std::snprintf(basename, sizeof(basename), "%s_sample_%06d.png", testname,
                sample);
  FilePath filename = directory.Append(basename);
  const int byte_count = width * height * 4;
  std::unique_ptr<char[]> pixels(new char[byte_count]);
  glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.get());
  write_png_file(filename.value().c_str(), pixels.get(), width, height);
}

int RequestedSamples() {
  const char* text = getenv("GLBENCH_CAPTURE_SAMPLE_COUNT");
  if (!text || !text[0])
    return 1;
  return static_cast<int>(strtol(text, nullptr, 10));
}

}  // namespace

int CapturedSampleCount() {
  return g_captured_samples;
}

double Bench(TestBase*) {
  return 0.0;
}

void RunTest(TestBase* test,
             const char* testname,
             double,
             const int width,
             const int height,
             bool) {
  const char* selected_case = getenv("GLBENCH_CAPTURE_CASE");
  if (!selected_case || std::strcmp(selected_case, testname) != 0)
    return;

  const int sample_count = RequestedSamples();
  for (int sample = 1; sample <= sample_count; ++sample) {
    char marker[32];
    std::snprintf(marker, sizeof(marker), "%d", sample);
    setenv("MESA_COUNTER_FRAME_TIME_MS", marker, 1);

    const bool renderdoc_started = StartRenderDocCapture();
    // The supported draw helpers issue one representative draw before their
    // iteration loop. Zero iterations therefore means one architecture
    // sample, without GLBench's adaptive performance-calibration traffic.
    if (!test->TestFunc(0)) {
      std::fprintf(stderr, "GLBench case failed: %s\n", testname);
      unsetenv("MESA_COUNTER_FRAME_TIME_MS");
      return;
    }
    glFinish();
    SaveSampleImage(testname, sample, width, height);
    EndRenderDocCapture(renderdoc_started);

    // Gallium's upstream HUD samples at frame boundaries. Pbuffer swap is the
    // portable end-of-frame signal and is harmless for captured pixels.
    g_main_gl_interface->SwapBuffers();
    unsetenv("MESA_COUNTER_FRAME_TIME_MS");
    glFinish();
    ++g_captured_samples;
    std::printf("@CAPTURE: %s sample=%d png=%s_sample_%06d.png\n", testname,
                sample, testname, sample);
    std::fflush(stdout);
  }
}

bool DrawArraysTestFunc::TestFunc(uint64_t iterations) {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  glFlush();
  for (uint64_t i = 0; i < iterations; ++i)
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  return true;
}

void DrawArraysTestFunc::FillRateTestNormal(const char* name) {
  FillRateTestNormalSubWindow(name, g_width, g_height);
}

void DrawArraysTestFunc::FillRateTestNormalSubWindow(const char* name,
                                                     const int width,
                                                     const int height) {
  RunTest(this, name, width * height, width, height, true);
}

void DrawArraysTestFunc::FillRateTestBlendDepth(const char* name) {
  char buffer[64];
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_BLEND);
  std::snprintf(buffer, sizeof(buffer), "%s_blended", name);
  RunTest(this, buffer, g_width * g_height, g_width, g_height, true);
  glDisable(GL_BLEND);

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_NOTEQUAL);
  std::snprintf(buffer, sizeof(buffer), "%s_depth_neq", name);
  RunTest(this, buffer, g_width * g_height, g_width, g_height, true);
  glDepthFunc(GL_NEVER);
  std::snprintf(buffer, sizeof(buffer), "%s_depth_never", name);
  RunTest(this, buffer, g_width * g_height, g_width, g_height, true);
  glDisable(GL_DEPTH_TEST);
}

bool DrawElementsTestFunc::TestFunc(uint64_t iterations) {
  glClearColor(0, 1.f, 0, 1.f);
  glClear(GL_COLOR_BUFFER_BIT);
  glDrawElements(GL_TRIANGLES, count_, GL_UNSIGNED_SHORT, nullptr);
  glFlush();
  for (uint64_t i = 0; i < iterations; ++i)
    glDrawElements(GL_TRIANGLES, count_, GL_UNSIGNED_SHORT, nullptr);
  return true;
}

}  // namespace glbench
