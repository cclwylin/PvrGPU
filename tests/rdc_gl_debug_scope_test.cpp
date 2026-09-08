#include "rdc_runner/gl_debug_scope.h"

#include <dlfcn.h>
#include <iostream>
#include <set>
#include <vector>

using namespace pvrgpu::rdc;
namespace {
constexpr unsigned Output = 0x92E0, Sync = 0x8242, Error = 0x824C;
constexpr unsigned High = 0x9146, Low = 0x9148, Note = 0x826B, Other = 0x8251;
unsigned checks = 0, previous_calls = 0;
const void *expected_user = &checks;
void Check(bool ok, const char *why) {
  ++checks;
  if (!ok) throw std::runtime_error(why);
}
template <typename F> void Reject(F f, const char *reason) {
  try { f(); } catch (const std::runtime_error &e) {
    Check(std::string(e.what()).find(reason) != std::string::npos, e.what());
    return;
  }
  Check(false, "expected named failure");
}
void Previous(unsigned, unsigned, unsigned, unsigned, int, const char *, const void *user) {
  Check(user == expected_user, "prior callback userdata preserved");
  ++previous_calls;
}
void Throwing(unsigned, unsigned, unsigned, unsigned, int, const char *, const void *) {
  throw std::runtime_error("prior callback threw");
}
struct Filter {
  bool all = false;
  std::set<unsigned> blocked{13, 29};
  bool operator==(const Filter &b) const { return all == b.all && blocked == b.blocked; }
};
struct Fake {
  void *context = this;
  unsigned error = 0;
  bool output = false, sync = false;
  int maximum = 64;
  GLDebugCallback callback = Previous;
  const void *user = expected_user;
  std::vector<Filter> filters{{}};
  static Fake *active;
  Fake() { active = this; }
  void Emit(unsigned type, unsigned severity, const char *message = "retained error", unsigned id = 17) {
    if (output && callback && (filters.back().all || !filters.back().blocked.count(id)))
      callback(0x8246, type, id, severity, -1, message, user);
  }
  GLDebugApi Api() {
    return {
      [] { return active->context; },
      [] { unsigned result = active->error; active->error = 0; return result; },
      [](unsigned key, int *value) { *value = key == 0x826D ? int(active->filters.size()) : active->maximum; },
      [](unsigned key, void **value) { *value = key == 0x8244 ? reinterpret_cast<void *>(active->callback) : const_cast<void *>(active->user); },
      [](unsigned key) -> unsigned char { return key == Output ? active->output : active->sync; },
      [](unsigned key) { (key == Output ? active->output : active->sync) = true; },
      [](unsigned key) { (key == Output ? active->output : active->sync) = false; },
      [](GLDebugCallback callback, const void *user) { active->callback = callback; active->user = user; },
      [](unsigned, unsigned, unsigned, int, const unsigned *, unsigned char enabled) { active->filters.back().all = enabled; },
      [](unsigned, unsigned, int, const char *) { active->filters.push_back(active->filters.back()); },
      [] { Check(active->filters.size() > 1, "no root-group underflow"); active->filters.pop_back(); }
    };
  }
};
Fake *Fake::active = nullptr;

void FakeTests() {
  for (unsigned flags = 0; flags < 4; ++flags) {
    Fake f;
    f.output = flags & 1; f.sync = flags & 2;
    f.filters.push_back({false, {7, 31, 101}});
    const auto saved = f.filters;
    const unsigned before = previous_calls;
    auto api = f.Api();
    {
      GLDebugScope scope(api);
      Check(f.output && f.sync && f.filters.back().all, "scope observes all synchronous messages");
      api.pushGroup(0x824A, 0, -1, "captured nested group");
      api.pushGroup(0x824A, 0, -1, "second captured group");
      f.Emit(Other, Low); f.Emit(Other, Note);
      api.popGroup(); api.popGroup();
      scope.Check("balanced groups");
      // Independently nested observers chain and restore without losing filters.
      { GLDebugScope nested(api); f.Emit(Other, Low); nested.Finish(); }
      scope.Finish();
    }
    Check(previous_calls == before + 3, "original callback is chained once per message");
    Check(f.callback == Previous && f.user == expected_user, "original callback restored");
    Check(f.output == bool(flags & 1) && f.sync == bool(flags & 2), "enable bits restored");
    Check(f.filters == saved, "complete nested filter stack restored");
  }
  for (auto pair : {std::pair{Error, Low}, std::pair{Other, High}, std::pair{Error, High}}) {
    Fake f;
    const auto saved = f.filters;
    auto api = f.Api();
    {
      GLDebugScope scope(api);
      f.error = 0x0500;
      f.Emit(pair.first, pair.second, "the original error was consumed", 13);
      Check(api.error() == 0x0500 && api.error() == 0, "simulate internal glGetError draining sticky flag");
      Reject([&] { scope.Check("after internal drain"); }, "original error was consumed");
      Reject([&] { scope.Finish(); }, "GL debug error");
    }
    Check(f.callback == Previous && f.filters == saved, "error unwind restores original observer and filters");
  }
  for (unsigned tamper = 0; tamper < 5; ++tamper) {
    Fake f;
    auto api = f.Api();
    {
      GLDebugScope scope(api);
      if (tamper == 0) f.output = false;
      if (tamper == 1) f.sync = false;
      if (tamper == 2) f.callback = Previous;
      if (tamper == 3) f.user = nullptr;
      if (tamper == 4) api.pushGroup(0x824A, 0, -1, "unclosed group");
      Reject([&] { scope.Check("tampered observer"); }, "replaced or disabled");
    }
    Check(f.callback == Previous && f.user == expected_user, "tamper failure restores callback");
    Check(f.filters.size() == 1, "unwind removes unclosed captured groups and independent scope");
  }
  {
    Fake f;
    auto api = f.Api();
    GLDebugScope scope(api);
    f.context = nullptr;
    Reject([&] { scope.Check("changed context"); }, "changed EGL context");
    f.context = &f; // restore for safe test cleanup; no scope may mutate a foreign context.
  }
  {
    Fake f; f.callback = Throwing;
    GLDebugScope scope(f.Api());
    f.Emit(Other, Low);
    Reject([&] { scope.Check("prior callback exception"); }, "GL debug error");
  }
  {
    Fake f;
    GLDebugScope scope(f.Api());
    const std::string long_message(4096, 'x');
    f.Emit(Error, High, long_message.c_str());
    Reject([&] { scope.Check("bounded error text"); }, "xxxxxxxx");
  }
  for (unsigned invalid = 0; invalid < 4; ++invalid) {
    Fake f; auto api = f.Api();
    if (invalid == 0) api.callback = nullptr;
    if (invalid == 1) f.context = nullptr;
    if (invalid == 2) f.maximum = 1;
    if (invalid == 3) f.error = 0x0500;
    Reject([&] { GLDebugScope scope(api); }, invalid == 0 ? "unavailable" :
           invalid == 1 ? "no replay EGL context" : invalid == 2 ? "capacity" : "GL error");
    Check(f.callback == Previous && f.filters.size() == 1, "failed constructor leaves prior state untouched");
  }
  for (const auto &name : {"glDebugMessageCallback", "glDebugMessageCallbackKHR", "glDebugMessageControl",
                           "MakeContextCurrent", "Internal::ImplicitThreadSwitch"}) {
    std::uint64_t groups = 0;
    Reject([&] { ValidateDebugCaptureCall(name, true, 0, groups); }, "Cannot guarantee");
  }
  for (const auto &name : {"glDisable", "glDisablei", "glDisableiEXT", "glDisableIndexedEXT", "glDisableiOES", "glDisableiNV"}) {
    for (auto cap : {Output, Sync}) {
      std::uint64_t groups = 0;
      Reject([&] { ValidateDebugCaptureCall(name, true, cap, groups); }, "can disable");
      Reject([&] { ValidateDebugCaptureCall(name, false, 0, groups); }, "can disable");
      ValidateDebugCaptureCall(name, true, 0x0B71, groups); // depth test does not affect observation.
    }
  }
  for (const auto &names : {std::pair{"glPushDebugGroup", "glPopDebugGroup"},
                            std::pair{"glPushGroupMarkerEXT", "glPopGroupMarkerEXT"}}) {
    std::uint64_t groups = 0;
    ValidateDebugCaptureCall(names.first, false, 0, groups);
    ValidateDebugCaptureCall(names.first, false, 0, groups);
    ValidateDebugCaptureCall(names.second, false, 0, groups);
    ValidateDebugCaptureCall(names.second, false, 0, groups);
    Check(groups == 0, "nested captured groups balance");
    Reject([&] { ValidateDebugCaptureCall(names.second, false, 0, groups); }, "escape");
    groups = UINT64_MAX;
    Reject([&] { ValidateDebugCaptureCall(names.first, false, 0, groups); }, "overflow");
  }
}

template <typename T> T Symbol(void *library, const char *name) {
  auto symbol = reinterpret_cast<T>(dlsym(library, name));
  Check(symbol != nullptr, name);
  return symbol;
}

void MesaTests(const char *egl_path, const char *gles_path) {
  void *egl = dlopen(egl_path, RTLD_NOW | RTLD_LOCAL), *gles = dlopen(gles_path, RTLD_NOW | RTLD_LOCAL);
  Check(egl && gles, "load explicitly selected Mesa libraries");
  auto display = Symbol<void *(*)(void *)>(egl, "eglGetDisplay")(nullptr);
  int major = 0, minor = 0;
  Check(Symbol<unsigned (*)(void *, int *, int *)>(egl, "eglInitialize")(display, &major, &minor), "initialize selected Mesa EGL");
  Check(Symbol<unsigned (*)(unsigned)>(egl, "eglBindAPI")(0x30A0), "bind GLES API");
  const int attrs[] = {0x3033, 1, 0x3040, 0x40, 0x3024, 8, 0x3023, 8, 0x3022, 8, 0x3038};
  void *config = nullptr; int count = 0;
  Check(Symbol<unsigned (*)(void *, const int *, void **, int, int *)>(egl, "eglChooseConfig")(display, attrs, &config, 1, &count) && count == 1, "GLES3 config");
  const int context_attrs[] = {0x3098, 3, 0x30FB, 2, 0x31B0, 1, 0x3038};
  const int surface_attrs[] = {0x3057, 1, 0x3056, 1, 0x3038};
  void *context = Symbol<void *(*)(void *, void *, void *, const int *)>(egl, "eglCreateContext")(display, config, nullptr, context_attrs);
  void *surface = Symbol<void *(*)(void *, void *, const int *)>(egl, "eglCreatePbufferSurface")(display, config, surface_attrs);
  auto make_current = Symbol<unsigned (*)(void *, void *, void *, void *)>(egl, "eglMakeCurrent");
  Check(context && surface && make_current(display, surface, surface, context), "debug GLES3.2 context current");
  GLDebugApi api;
  api.current = Symbol<decltype(api.current)>(egl, "eglGetCurrentContext");
#define LOAD(member, name) api.member = Symbol<decltype(api.member)>(gles, name)
  LOAD(error, "glGetError"); LOAD(get, "glGetIntegerv"); LOAD(getPointer, "glGetPointerv");
  LOAD(isEnabled, "glIsEnabled"); LOAD(enable, "glEnable"); LOAD(disable, "glDisable");
  LOAD(callback, "glDebugMessageCallback"); LOAD(control, "glDebugMessageControl");
  LOAD(pushGroup, "glPushDebugGroup"); LOAD(popGroup, "glPopDebugGroup");
#undef LOAD
  auto insert = Symbol<void (*)(unsigned, unsigned, unsigned, unsigned, int, const char *)>(gles, "glDebugMessageInsert");
  api.enable(Output); api.enable(Sync); api.callback(Previous, expected_user);
  api.pushGroup(0x824A, 77, -1, "original outer group");
  api.control(0x1100, 0x1100, 0x1100, 0, nullptr, 0);
  const unsigned allowed = 42;
  api.control(0x824A, Other, 0x1100, 1, &allowed, 1);
  auto filter_check = [&] {
    const unsigned before = previous_calls;
    insert(0x824A, Other, 42, Low, -1, "enabled original ID");
    insert(0x824A, Other, 43, Low, -1, "disabled original ID");
    Check(previous_calls == before + 1, "exact original per-ID filter restored");
    void *callback = nullptr, *user = nullptr;
    api.getPointer(0x8244, &callback); api.getPointer(0x8245, &user);
    int depth = 0; api.get(0x826D, &depth);
    Check(callback == reinterpret_cast<void *>(Previous) && user == expected_user && depth == 2,
          "Mesa original callback, userdata and nested group restored");
    Check(api.error() == 0, "state verification uses valid GL operations");
  };
  filter_check();
  {
    GLDebugScope scope(api);
    api.pushGroup(0x824A, 91, -1, "captured group");
    api.pushGroup(0x824A, 92, -1, "nested captured group");
    insert(0x824A, Other, 43, Low, -1, "observer enables previously disabled message");
    api.popGroup(); api.popGroup();
    scope.Finish();
  }
  filter_check();
  {
    GLDebugScope scope(api);
    api.enable(0xFFFFFFFFU); // genuine GL_INVALID_ENUM; this is not a synthetic callback message.
    Check(api.error() == 0x0500 && api.error() == 0, "real Mesa error consumed by internal glGetError");
    Reject([&] { scope.Check("drained genuine Mesa error"); }, "GL debug error");
  }
  filter_check();
  {
    GLDebugScope scope(api);
    insert(0x824A, Other, 99, High, -1, "high severity without sticky GL error");
    Check(api.error() == 0, "high severity alone need not set glGetError");
    Reject([&] { scope.Finish(); }, "high severity without sticky GL error");
  }
  filter_check();
  api.popGroup(); api.callback(nullptr, nullptr);
  Check(make_current(display, nullptr, nullptr, nullptr), "release EGL current");
  Check(Symbol<unsigned (*)(void *, void *)>(egl, "eglDestroySurface")(display, surface), "destroy surface");
  Check(Symbol<unsigned (*)(void *, void *)>(egl, "eglDestroyContext")(display, context), "destroy context");
  Check(Symbol<unsigned (*)(void *)>(egl, "eglTerminate")(display), "terminate display");
  dlclose(gles); dlclose(egl);
}
} // namespace

int main(int argc, char **argv) {
  try {
    Check(argc == 1 || argc == 3, "usage: rdc-gl-debug-scope-test [MESA_EGL_LIBRARY MESA_GLES_LIBRARY]");
    FakeTests();
    if (argc == 3) MesaTests(argv[1], argv[2]);
    std::cout << "GL debug scope " << (argc == 3 ? "fake + real Mesa" : "fake") << " PASS: " << checks << " checks\n";
    return 0;
  } catch (const std::exception &e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
