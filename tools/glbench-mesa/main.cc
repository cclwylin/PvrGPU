// Capture-oriented entry point for official ChromeOS GLBench workloads.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include "all_tests.h"
#include "glinterface.h"
#include "main.h"
#include "testbase.h"
#include "utils.h"

bool g_verbose = false;
bool g_hasty = true;
bool g_notemp = true;
GLint g_max_texture_size = 0;

namespace glbench {
int CapturedSampleCount();
}

namespace {

struct Options {
  std::string test = "fill_rate";
  std::string test_case = "fill_tex_bilinear";
  std::string outdir;
  int samples = 1;
  int width = WINDOW_WIDTH;
  int height = WINDOW_HEIGHT;
  bool list = false;
};

void Usage(const char* argv0) {
  std::printf(
      "Usage: %s [--test GROUP] [--case CASE] [--sample N] "
      "[--size WIDTHxHEIGHT] [--outdir DIR]\n",
      argv0);
}

void ListCases() {
  std::puts("fill_rate:fill_solid");
  std::puts("fill_rate:fill_solid_blended");
  std::puts("fill_rate:fill_solid_depth_neq");
  std::puts("fill_rate:fill_solid_depth_never");
  std::puts("fill_rate:fill_tex_nearest");
  std::puts("fill_rate:fill_tex_bilinear");
  std::puts("fill_rate:fill_tex_trilinear_linear_05");
  std::puts("fill_rate:fill_tex_trilinear_linear_04");
  std::puts("fill_rate:fill_tex_trilinear_linear_01");
  std::puts("triangle_setup:triangle_setup");
  std::puts("triangle_setup:triangle_setup_all_culled");
  std::puts("triangle_setup:triangle_setup_half_culled");
  std::puts("attribute_fetch_shader:attribute_fetch_shader");
  std::puts("attribute_fetch_shader:attribute_fetch_shader_2_attr");
  std::puts("attribute_fetch_shader:attribute_fetch_shader_4_attr");
  std::puts("attribute_fetch_shader:attribute_fetch_shader_8_attr");
  std::puts("varyings_ddx_shader:varyings_shader_1");
  std::puts("varyings_ddx_shader:varyings_shader_2");
  std::puts("varyings_ddx_shader:varyings_shader_4");
  std::puts("varyings_ddx_shader:varyings_shader_8");
}

bool ParsePositiveInt(const char* text, int* value) {
  char* end = nullptr;
  long parsed = std::strtol(text, &end, 10);
  if (!text[0] || !end || end[0] || parsed <= 0 || parsed > 100000)
    return false;
  *value = static_cast<int>(parsed);
  return true;
}

bool ParseOptions(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--list" || arg == "-list") {
      options->list = true;
      continue;
    }
    if (arg == "--help" || arg == "-h") {
      Usage(argv[0]);
      std::exit(0);
    }
    if (i + 1 >= argc) {
      std::fprintf(stderr, "Missing value after %s.\n", argv[i]);
      return false;
    }
    const char* value = argv[++i];
    if (arg == "--test" || arg == "-test") {
      options->test = value;
    } else if (arg == "--case" || arg == "-case") {
      options->test_case = value;
    } else if (arg == "--sample" || arg == "-sample") {
      if (!ParsePositiveInt(value, &options->samples))
        return false;
    } else if (arg == "--outdir" || arg == "-outdir") {
      options->outdir = value;
    } else if (arg == "--size" || arg == "-size") {
      int consumed = 0;
      if (std::sscanf(value, "%dx%d%n", &options->width, &options->height,
                      &consumed) != 2 ||
          consumed != static_cast<int>(std::strlen(value)) ||
          options->width <= 0 || options->height <= 0) {
        return false;
      }
    } else {
      std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
      return false;
    }
  }
  return true;
}

glbench::TestBase* CreateTest(const std::string& name) {
  if (name == "fill_rate")
    return glbench::GetFillRateTest();
  if (name == "triangle_setup")
    return glbench::GetTriangleSetupTest();
  if (name == "attribute_fetch_shader")
    return glbench::GetAttributeFetchShaderTest();
  if (name == "varyings_ddx_shader")
    return glbench::GetVaryingsAndDdxyShaderTest();
  return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseOptions(argc, argv, &options)) {
    Usage(argv[0]);
    return 2;
  }
  if (options.list) {
    ListCases();
    return 0;
  }
  if (options.outdir.empty()) {
    std::fprintf(stderr, "--outdir is required for a workload run.\n");
    Usage(argv[0]);
    return 2;
  }

  std::unique_ptr<glbench::TestBase> test(CreateTest(options.test));
  if (!test) {
    std::fprintf(stderr, "Unsupported GLBench test group: %s\n",
                 options.test.c_str());
    return 2;
  }

  g_width = options.width;
  g_height = options.height;
  setenv("GLBENCH_CAPTURE_CASE", options.test_case.c_str(), 1);
  setenv("GLBENCH_CAPTURE_OUTPUT_DIR", options.outdir.c_str(), 1);
  const std::string sample_count = std::to_string(options.samples);
  setenv("GLBENCH_CAPTURE_SAMPLE_COUNT", sample_count.c_str(), 1);

  g_main_gl_interface.reset(GLInterface::Create());
  if (!g_main_gl_interface || !g_main_gl_interface->Init()) {
    std::fprintf(stderr, "Failed to initialize Mesa surfaceless EGL.\n");
    return 1;
  }

  std::printf("# board_id: %s - %s\n", glGetString(GL_VENDOR),
              glGetString(GL_RENDERER));
  GLint viewport_dims[2] = {0, 0};
  glGetIntegerv(GL_MAX_VIEWPORT_DIMS, viewport_dims);
  glGetIntegerv(GL_MAX_TEXTURE_SIZE, &g_max_texture_size);
  if (viewport_dims[0] < g_width || viewport_dims[1] < g_height ||
      g_max_texture_size < g_width || g_max_texture_size < g_height) {
    std::fprintf(stderr, "Requested surface exceeds Mesa GL limits.\n");
    g_main_gl_interface->Cleanup();
    return 1;
  }

  std::printf("# GL_VERSION: %s\n", glGetString(GL_VERSION));
  std::printf("# GLBench group: %s\n", options.test.c_str());
  std::printf("# GLBench case: %s\n", options.test_case.c_str());
  glbench::ClearBuffers();
  const bool run_ok = test->Run();
  glFinish();
  g_main_gl_interface->Cleanup();

  if (!run_ok || glbench::CapturedSampleCount() != options.samples) {
    std::fprintf(stderr,
                 "GLBench case '%s' was not captured; check the group/case "
                 "pair.\n",
                 options.test_case.c_str());
    return 1;
  }

  std::puts("@TEST_END");
  return 0;
}
