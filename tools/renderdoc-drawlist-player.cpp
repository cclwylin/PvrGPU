// Regular DrawList checkpointing: restore persistent replay state in a fresh
// controller, then execute the actual ordered suffix, never a state-only prefix.
// Requires the explicit versioned RenderDoc snapshot extension in third_party/.
#include "api/replay/renderdoc_replay.h"
#include "rdc_runner/gl_debug_scope.h"
#include "rdc_runner/replay_range_audit.h"
#include "rdc_runner/sha256.h"
#include "drawlist-color-readback.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

REPLAY_PROGRAM_MARKER()
namespace fs = std::filesystem;
namespace {
void require(bool value, const std::string &why)
{ if(!value) throw std::runtime_error(why); }

std::string quote(const std::string &value)
{
  std::ostringstream out;
  out << '"';
  for(unsigned char c : value)
    if(c == '"' || c == '\\') out << '\\' << c;
    else if(c < 32) out << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                        << unsigned(c) << std::dec;
    else out << c;
  return out.str() + '"';
}

uint32_t number(const std::string &text)
{
  require(!text.empty() && text.find_first_not_of("0123456789") == std::string::npos,
          "Expected an unsigned decimal integer: " + text);
  const auto result = std::stoull(text);
  require(result <= UINT32_MAX, "Integer exceeds replay event range");
  return uint32_t(result);
}

std::string hashFile(const fs::path &path)
{
  std::string digest, error;
  require(pvrgpu::rdc::Sha256File(path, &digest, &error), error);
  return digest;
}

fs::path checkedPath(const fs::path &input, bool existing)
{
  require(!input.empty(), "Empty path");
  const fs::path path = fs::absolute(input);
  fs::path walk;
  for(const auto &part : path)
  {
    require(part != "..", "Parent traversal is not allowed: " + path.string());
    walk /= part;
    require(!fs::is_symlink(fs::symlink_status(walk)), "Symlink is not allowed: " + walk.string());
  }
  require(existing ? fs::is_regular_file(path) : !fs::exists(path),
          existing ? "Input file missing: " + path.string() : "Output already exists: " + path.string());
  if(!existing) require(fs::is_directory(path.parent_path()), "Output parent must exist: " + path.string());
  return path.lexically_normal();
}

void exclusiveWrite(const fs::path &path, const std::string &bytes)
{
  int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
  require(fd >= 0, "Cannot exclusively create " + path.string());
  size_t done = 0;
  while(done < bytes.size())
  {
    const ssize_t written = write(fd, bytes.data() + done, bytes.size() - done);
    if(written < 0 && errno == EINTR) continue;
    if(written <= 0) { close(fd); throw std::runtime_error("Could not finish " + path.string()); }
    done += size_t(written);
  }
  const bool synced = fsync(fd) == 0;
  const bool closed = close(fd) == 0;
  require(synced && closed, "Could not synchronize " + path.string());
}

std::string readLog(const fs::path &path)
{
  // A zero-work suffix legitimately produces no model JSONL file.
  if(!fs::exists(path)) return {};
  std::ifstream in(path, std::ios::binary);
  require(bool(in), "Cannot read audit log " + path.string());
  std::string result{std::istreambuf_iterator<char>(in), {}};
  require(!in.bad(), "Could not finish audit log " + path.string());
  return result;
}

std::string environment(const char *name)
{
  const char *value = std::getenv(name);
  require(value && *value, std::string("Required environment variable: ") + name);
  return value;
}

struct Arguments
{
  fs::path capture, output, receipt, input, color;
  uint32_t draw = 0, expected = 0;
  bool resume = false;
  Arguments(int argc, char **argv)
  {
    require(argc >= 2, "Missing capture argument; see --help");
    capture = checkedPath(argv[1], true);
    bool haveDraw = false, haveExpected = false;
    for(int i = 2; i < argc; i += 2)
    {
      require(i + 1 < argc, "Missing option value");
      const std::string flag = argv[i], value = argv[i + 1];
      if(flag == "--stop-after-draw") { require(!haveDraw, "Duplicate draw option"); draw = number(value); haveDraw = true; }
      else if(flag == "--expected-resume-event") { require(!haveExpected, "Duplicate expected event"); expected = number(value); haveExpected = true; }
      else if(flag == "--state-out") { require(output.empty(), "Duplicate state output"); output = checkedPath(value, false); }
      else if(flag == "--receipt-out") { require(receipt.empty(), "Duplicate receipt output"); receipt = checkedPath(value, false); }
      else if(flag == "--state-in") { require(input.empty(), "Duplicate state input"); input = checkedPath(value, true); }
      else if(flag == "--color-out") { require(color.empty(), "Duplicate color output"); color = checkedPath(value, false); }
      else throw std::runtime_error("Unknown option: " + flag);
    }
    resume = !input.empty();
    require(haveDraw && !output.empty() && !receipt.empty(), "Draw, state output and receipt output are required");
    require(resume == haveExpected && (!resume || expected > 0), "Resume requires a positive expected event");
    require(output != receipt && capture != output && capture != receipt &&
            (!resume || (input != output && input != receipt && input != capture)), "Input/output paths must be distinct");
  }
};

struct SnapshotApi
{
  using Version = uint32_t (*)();
  using Save = int (*)(IReplayController *, uint32_t, const char *, const char *, char *, size_t);
  using Load = int (*)(IReplayController *, const char *, const char *, uint32_t *, char *, size_t);
  void *library = nullptr;
  Save save = nullptr;
  Load load = nullptr;
  fs::path path;
  template<typename T> T symbol(const char *name)
  {
    void *address = dlsym(library, name);
    require(address != nullptr, std::string("Snapshot extension missing export: ") + name);
    Dl_info info{};
    require(dladdr(address, &info) != 0 && info.dli_fname && fs::canonical(info.dli_fname) == path,
            std::string("Snapshot export came from a different library: ") + name);
    return reinterpret_cast<T>(address);
  }
  SnapshotApi()
  {
    path = fs::canonical(environment("PVRGPU_RENDERDOC_LIB"));
    Dl_info linked{};
    require(dladdr(reinterpret_cast<void *>(&RENDERDOC_OpenCaptureFile), &linked) != 0 && linked.dli_fname,
            "Cannot identify linked RenderDoc library");
    require(fs::canonical(linked.dli_fname) == path,
            "Player is linked to a different RenderDoc library; rebuild it for --renderdoc-lib");
    library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    require(library != nullptr, "Cannot load selected RenderDoc library");
    require(symbol<Version>("RENDERDOC_GetReplaySnapshotVersion")() == 1, "Unsupported RenderDoc snapshot ABI");
    save = symbol<Save>("RENDERDOC_SaveReplaySnapshot");
    load = symbol<Load>("RENDERDOC_LoadReplaySnapshot");
  }
  ~SnapshotApi() { if(library) dlclose(library); }
};

struct NativeScope
{
  struct Entry { const char *key; bool present; std::string value; };
  std::vector<Entry> entries;
  NativeScope()
  {
    for(const char *key : {"PVRGPU_SYSTEMC_API_LIB", "PVRGPU_SYSTEMC_BRIDGE", "PVRGPU_DRIVER_COMMAND_OUT",
                          "PVRGPU_SYSTEMC_JSONL_OUT", "PVRGPU_DRIVER_COUNTER_OUT", "MESA_COUNTER_FRAME_TIME_MS"})
    {
      const char *value = std::getenv(key);
      entries.push_back({key, value != nullptr, value ? value : ""});
    }
    disable();
  }
  void disable() const { for(const auto &entry : entries) unsetenv(entry.key); }
  void enable() const
  {
    for(const auto &entry : entries) if(entry.present) setenv(entry.key, entry.value.c_str(), 1);
    setenv("MESA_COUNTER_FRAME_TIME_MS", "0", 1);
  }
  ~NativeScope() { disable(); }
};

struct ReplayLifetime
{
  ICaptureFile *file = nullptr;
  IReplayController *controller = nullptr;
  ReplayLifetime()
  {
    GlobalEnvironment env; env.enumerateGPUs = false;
    RENDERDOC_InitialiseReplay(env, {});
  }
  ~ReplayLifetime()
  {
    if(controller) controller->Shutdown();
    if(file) file->Shutdown();
    RENDERDOC_ShutdownReplay();
  }
};

struct GL
{
  pvrgpu::rdc::GLDebugApi debug;
  void *egl = nullptr, *gles = nullptr;
  void (*finish)() = nullptr;
  template<typename T> static T symbol(void *library, const char *name)
  {
    auto result = reinterpret_cast<T>(dlsym(library, name));
    require(result != nullptr, std::string("Missing Mesa GL entry point: ") + name);
    return result;
  }
  GL()
  {
    egl = dlopen(environment("RENDERDOC_MESA_EGL_PATH").c_str(), RTLD_NOW | RTLD_LOCAL);
    gles = dlopen(environment("RENDERDOC_MESA_GLES_PATH").c_str(), RTLD_NOW | RTLD_LOCAL);
    require(egl && gles, "Cannot load explicit Mesa EGL/GLES libraries");
    finish = symbol<decltype(finish)>(gles, "glFinish");
    debug.current = symbol<decltype(debug.current)>(egl, "eglGetCurrentContext");
#define LOAD(member, name) debug.member = symbol<decltype(debug.member)>(gles, name)
    LOAD(error, "glGetError"); LOAD(get, "glGetIntegerv"); LOAD(getPointer, "glGetPointerv");
    LOAD(isEnabled, "glIsEnabled"); LOAD(enable, "glEnable"); LOAD(disable, "glDisable");
    LOAD(callback, "glDebugMessageCallback"); LOAD(control, "glDebugMessageControl");
    LOAD(pushGroup, "glPushDebugGroup"); LOAD(popGroup, "glPopDebugGroup");
#undef LOAD
  }
  ~GL() { if(gles) dlclose(gles); if(egl) dlclose(egl); }
};

void collectActions(const rdcarray<ActionDescription> &actions, std::vector<uint32_t> &draws, uint32_t &last)
{
  for(const auto &action : actions)
  {
    last = std::max(last, action.eventId);
    // A multi-draw parent is a marker; its actual leaf draws retain DrawList IDs.
    if(action.children.empty() && (action.flags & ActionFlags::Drawcall) != ActionFlags::NoFlags)
      draws.push_back(action.eventId);
    collectActions(action.children, draws, last);
  }
}

void validateDebugObservation(const SDFile &file)
{
  uint64_t groups = 0;
  for(const SDChunk *chunk : file.chunks)
  {
    if(!chunk) continue;
    bool hasCap = false; uint64_t cap = 0;
    for(size_t i = 0; i < chunk->NumChildren(); ++i)
    {
      const SDObject *child = chunk->GetChild(i);
      if(child && child->name == "cap" && (child->type.basetype == SDBasic::Enum ||
                                           child->type.basetype == SDBasic::UnsignedInteger))
      { hasCap = true; cap = child->data.basic.u; }
    }
    pvrgpu::rdc::ValidateDebugCaptureCall(chunk->name.c_str(), hasCap, cap, groups);
  }
}

void validateReplay(IReplayController *controller, const char *phase)
{
  require(controller->GetFatalErrorStatus().code == ResultCode::Succeeded,
          std::string(phase) + ": RenderDoc reported a fatal error");
  for(const auto &message : controller->GetDebugMessages())
    require(message.severity != MessageSeverity::High && message.source != MessageSource::IncorrectAPIUse &&
            message.source != MessageSource::UnsupportedConfiguration,
            std::string(phase) + ": validation error at event " + std::to_string(message.eventId) + ": " + message.description.c_str());
}
} // namespace

int main(int argc, char **argv)
{
  if(argc == 2 && std::string(argv[1]) == "--help")
  {
    std::cout << "Usage: " << argv[0] << " CAPTURE --stop-after-draw N --state-out FILE --receipt-out FILE\n"
              << "       [--state-in FILE --expected-resume-event E] [--color-out FILE]\n"
              << "DrawList N is zero-based and inclusive. Each invocation saves full functional state.\n"
              << "Use script/run_drawlist_replay.py for runtime checks and a committed manifest.\n";
    return 0;
  }
  try
  {
    const Arguments args(argc, argv);
    const auto captureHash = hashFile(args.capture);
    const std::string backend = environment("GALLIUM_DRIVER");
    require(backend == "pvrgpu" || backend == "llvmpipe", "Only pvrgpu and llvmpipe backends are supported");
    const bool nativeBackend = backend == "pvrgpu";
    fs::path driverLog, modelLog, commandLog;
    const fs::path restoreLog = args.receipt.string() + ".restore-driver.txt";
    const fs::path saveLog = args.receipt.string() + ".save-driver.txt";
    std::set<fs::path> paths{args.capture};
    if(args.resume) paths.insert(args.input);
    const auto addOutput = [&](const fs::path &path) {
      if(path.empty()) return;
      const auto checked = checkedPath(path, false);
      require(paths.insert(checked).second, "All input, output and audit paths must be distinct: " + checked.string());
    };
    addOutput(args.output); addOutput(args.receipt); addOutput(args.color);
    if(nativeBackend)
    {
      require(fs::is_regular_file(environment("PVRGPU_SYSTEMC_API_LIB")), "Native bridge file is required");
      driverLog = checkedPath(environment("PVRGPU_DRIVER_COUNTER_OUT"), false);
      modelLog = checkedPath(environment("PVRGPU_SYSTEMC_JSONL_OUT"), false);
      commandLog = checkedPath(environment("PVRGPU_DRIVER_COMMAND_OUT"), false);
      addOutput(driverLog); addOutput(modelLog); addOutput(commandLog);
      addOutput(restoreLog); addOutput(saveLog);
    }
    const SnapshotApi snapshot;
    NativeScope native;
    unsetenv("PVRGPU_RDC_TRACE_DRAW_ACTIONS"); unsetenv("PVRGPU_RDC_CASE_NAME");
    ReplayLifetime replay;
    struct DisableBeforeShutdown { NativeScope &scope; ~DisableBeforeShutdown() { scope.disable(); } } disable{native};
    std::cerr << "Opening capture with native execution isolated (not replay evidence)\n";
    replay.file = RENDERDOC_OpenCaptureFile();
    require(replay.file != nullptr, "Cannot create capture reader");
    require(replay.file->OpenFile(args.capture.c_str(), "rdc", nullptr).code == ResultCode::Succeeded, "OpenFile failed");
    ReplayOptions options; options.apiValidation = true;
    const auto opened = replay.file->OpenCapture(options, nullptr);
    replay.controller = opened.second;
    require(opened.first.code == ResultCode::Succeeded && replay.controller, "OpenCapture failed");
    std::vector<uint32_t> draws; uint32_t lastEvent = 0;
    collectActions(replay.controller->GetRootActions(), draws, lastEvent);
    require(args.draw < draws.size(), "Requested DrawList is outside capture");
    require(std::is_sorted(draws.begin(), draws.end()) && std::adjacent_find(draws.begin(), draws.end()) == draws.end(),
            "Capture does not have strictly ordered unique DrawList events");
    const uint32_t stop = args.draw + 1 == draws.size() ? lastEvent : draws[args.draw];
    require(stop && stop < UINT32_MAX, "Invalid replay stop event");
    if(args.resume)
    {
      const auto previous = std::find(draws.begin(), draws.end(), args.expected);
      require(previous != draws.end() && size_t(previous - draws.begin()) < args.draw && args.expected < stop,
              "Resume event is not a preceding nonterminal DrawList boundary");
    }
    validateDebugObservation(replay.controller->GetStructuredFile());
    GL gl;
    // Only bootstrap errors may be drained. Restoration and every actual event
    // below use the independent synchronous observer, even if RenderDoc drains GL.
    for(unsigned i = 0; i < 64 && gl.debug.error(); ++i) {}
    require(gl.debug.error() == 0, "Cannot clear bootstrap GL diagnostics");
    replay.controller->GetDebugMessages();
    if(nativeBackend)
    {
      exclusiveWrite(restoreLog, {});
      setenv("PVRGPU_DRIVER_COUNTER_OUT", restoreLog.c_str(), 1);
    }
    uint32_t resumed = 0;
    std::array<char, 4096> error{};
    std::cerr << (args.resume ? "Loading and verifying complete checkpoint state\n" : "Restoring original capture initial state\n");
    {
      pvrgpu::rdc::GLDebugScope observation(gl.debug);
      if(args.resume)
      {
        const int code = snapshot.load(replay.controller, args.input.c_str(), captureHash.c_str(), &resumed, error.data(), error.size());
        require(code == 0, "Snapshot load refused: " + std::string(error.data()));
        require(resumed == args.expected, "Snapshot restored a different event");
      }
      else replay.controller->SetFrameEvent(0, true);
      gl.finish();
      observation.Finish();
    }
    validateReplay(replay.controller, "Persistent state restoration");
    if(nativeBackend)
    {
      unsetenv("PVRGPU_DRIVER_COUNTER_OUT");
      const auto bytes = readLog(restoreLog);
      const auto audit = pvrgpu::rdc::AuditNativeDriverEvents(bytes, true);
      require(audit.empty(), "Snapshot restoration: " + audit);
      if(!args.resume)
      {
        std::string why;
        require(pvrgpu::rdc::ValidateInitialCopyAudit(bytes, &why), why);
      }
    }
    void *context = gl.debug.current();
    require(context != nullptr, "Restoration left no replay EGL context");
    native.enable();
    std::cerr << "Executing ordered events " << resumed + 1 << ".." << stop
              << " through DrawList " << args.draw << '\n';
    std::string colorJson = "null";
    {
      pvrgpu::rdc::GLDebugScope observation(gl.debug);
      require(replay.controller->ReplayEventRange(resumed + 1, stop), "Ordered replay range failed");
      require(gl.debug.current() == context, "Replay changed EGL context");
      gl.finish();
      if(!args.color.empty())
        colorJson = pvrgpu::rdc::ReadCompletedDrawColor(gl.gles, args.color).ToJson();
      observation.Finish();
    }
    validateReplay(replay.controller, "Completed ordered replay");
    native.disable();
    if(nativeBackend)
    {
      require(fs::is_regular_file(driverLog), "Replay produced no driver audit");
      const auto audit = pvrgpu::rdc::AuditNativeReplayRange(readLog(modelLog), readLog(driverLog));
      require(audit.ok, "Native replay audit failed: " + audit.error);
    }
    // Save cannot execute shaders to produce missing state. It may only perform
    // bindings, synchronization, and lossless raw resource transfers.
    std::cerr << "Completed replay audit; saving synchronized functional state\n";
    if(nativeBackend)
    {
      exclusiveWrite(saveLog, {});
      setenv("PVRGPU_DRIVER_COUNTER_OUT", saveLog.c_str(), 1);
    }
    {
      pvrgpu::rdc::GLDebugScope observation(gl.debug);
      error.fill(0);
      const int code = snapshot.save(replay.controller, stop, args.output.c_str(), captureHash.c_str(), error.data(), error.size());
      require(code == 0, "Snapshot save refused: " + std::string(error.data()));
      require(gl.debug.current() == context, "Snapshot save changed EGL context");
      gl.finish();
      observation.Finish();
    }
    validateReplay(replay.controller, "Snapshot serialization");
    native.disable();
    if(nativeBackend)
    {
      const auto audit = pvrgpu::rdc::AuditNativeDriverEvents(readLog(saveLog), true);
      require(audit.empty(), "Snapshot serialization: " + audit);
    }
    require(fs::is_regular_file(args.output) && fs::file_size(args.output) > 0, "Snapshot extension produced no state file");
    require(hashFile(args.capture) == captureHash, "Source capture changed during replay");
    std::ostringstream receipt;
    receipt << "{\"schema\":\"pvrgpu.drawlist-replay.v1\",\"status\":\"snapshot_saved\",\"backend\":" << quote(backend)
            << ",\"capture_path\":" << quote(args.capture.string()) << ",\"state_path\":" << quote(args.output.string())
            << ",\"source_capture_sha256\":" << quote(captureHash) << ",\"after_draw\":" << args.draw
            << ",\"after_event\":" << stop << ",\"next_event\":" << (stop == lastEvent ? "null" : std::to_string(stop + 1))
            << ",\"capture_last_event\":" << lastEvent << ",\"trace_draw_actions\":" << draws.size()
            << ",\"resumed_from_event\":" << (args.resume ? std::to_string(resumed) : "null")
            << ",\"native_prefix_replayed\":false,\"snapshot_state_sha256\":" << quote(hashFile(args.output))
            << ",\"snapshot_api_version\":1,\"context_finished\":true,\"api_errors\":0"
            << ",\"snapshot_restore_verified\":" << (args.resume ? "true" : "false")
            << ",\"cold_cache\":true,\"color_output\":" << colorJson << "}\n";
    exclusiveWrite(args.receipt, receipt.str());
    std::cout << "Saved DrawList " << args.draw << " after event " << stop << ": " << args.output << '\n';
    return 0;
  }
  catch(const std::exception &error)
  {
    std::cerr << "DrawList replay failed: " << error.what() << '\n';
    return 1;
  }
}
