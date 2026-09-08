// One complete, ordered OpenGL replay and an actual completed-FBO PNG.
// This uses the pinned RenderDoc public ReplayEventRange API, never a private
// range library. A single range does not split live query/transform-feedback
// scopes. Initialization is explicitly NOT claimed as native shader execution.
#include "api/replay/renderdoc_replay.h"
#include "support/png_writer.h"
#include "rdc_runner/gl_debug_scope.h"
#include "rdc_runner/native_report.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

REPLAY_PROGRAM_MARKER()
namespace fs = std::filesystem;

static void require(bool condition, const std::string &why)
{
  if(!condition) throw std::runtime_error(why);
}

static std::string json(const std::string &value)
{
  std::ostringstream out;
  out << '"';
  for(unsigned char c : value)
  {
    if(c == '"' || c == '\\') out << '\\' << c;
    else if(c < 0x20) out << "\\u00" << std::hex << std::setw(2)
                           << std::setfill('0') << unsigned(c) << std::dec;
    else out << c;
  }
  out << '"';
  return out.str();
}

static std::string resourceString(const ResourceId &resource)
{
  // The pinned API's opaque ID is one uint64. Copy its representation without
  // creating/modifying an ID or relying on unexported DoStringise templates.
  static_assert(sizeof(ResourceId) == sizeof(uint64_t), "RenderDoc resource-ID ABI changed");
  uint64_t id = 0;
  std::memcpy(&id, &resource, sizeof(id));
  return "ResourceId::" + std::to_string(id);
}

struct NativeScope
{
  struct Entry { const char *key; bool present; std::string value; };
  std::vector<Entry> entries;
  NativeScope()
  {
    // COMMAND has a JSONL fallback; clearing just COMMAND does not isolate init.
    // The counter path must also be absent: init compiler refusals are not
    // failures in the subsequent measured native replay.
    for(const char *key : {"PVRGPU_SYSTEMC_API_LIB", "PVRGPU_SYSTEMC_BRIDGE",
                          "PVRGPU_DRIVER_COMMAND_OUT", "PVRGPU_SYSTEMC_JSONL_OUT",
                          "PVRGPU_DRIVER_COUNTER_OUT", "MESA_COUNTER_FRAME_TIME_MS"})
    {
      const char *value = std::getenv(key);
      entries.push_back({key, value != nullptr, value ? value : ""});
    }
    disable();
  }
  void disable() const { for(const auto &entry : entries) unsetenv(entry.key); }
  void enable() const
  {
    for(const auto &entry : entries)
      if(entry.present) setenv(entry.key, entry.value.c_str(), 1);
    // A marker selects the real full-frame work, not a workload-specific count.
    setenv("MESA_COUNTER_FRAME_TIME_MS", "0", 1);
  }
  ~NativeScope() { disable(); }
};

struct ReplayLifetime
{
  ICaptureFile *file = nullptr;
  IReplayController *renderer = nullptr;
  ReplayLifetime()
  {
    GlobalEnvironment environment;
    environment.enumerateGPUs = false;
    RENDERDOC_InitialiseReplay(environment, {});
  }
  ~ReplayLifetime()
  {
    if(renderer) renderer->Shutdown();
    if(file) file->Shutdown();
    RENDERDOC_ShutdownReplay();
  }
};

struct GL
{
  pvrgpu::rdc::GLDebugApi debug;
  void *egl = nullptr, *gles = nullptr;
  void *(*current)() = nullptr;
  void (*finish)() = nullptr;
  unsigned (*error)() = nullptr;
  void (*get)(unsigned, int *) = nullptr;
  void (*attachment)(unsigned, unsigned, unsigned, int *) = nullptr;
  void (*bindFramebuffer)(unsigned, unsigned) = nullptr;
  void (*bindBuffer)(unsigned, unsigned) = nullptr;
  void (*bindTexture)(unsigned, unsigned) = nullptr;
  void (*bindRenderbuffer)(unsigned, unsigned) = nullptr;
  void (*texLevel)(unsigned, int, unsigned, int *) = nullptr;
  void (*renderbuffer)(unsigned, unsigned, int *) = nullptr;
  void (*pack)(unsigned, int) = nullptr;
  void (*readBuffer)(unsigned) = nullptr;
  void (*readPixels)(int, int, int, int, unsigned, unsigned, void *) = nullptr;
  const unsigned char *(*stringIndex)(unsigned, unsigned) = nullptr;
  template <typename T> static T symbol(void *library, const char *name)
  {
    auto result = reinterpret_cast<T>(dlsym(library, name));
    require(result != nullptr, std::string("Missing Mesa GL entry point: ") + name);
    return result;
  }
  GL()
  {
    const char *eglPath = std::getenv("RENDERDOC_MESA_EGL_PATH");
    const char *glesPath = std::getenv("RENDERDOC_MESA_GLES_PATH");
    require(eglPath && glesPath, "Explicit Mesa EGL/GLES library paths are required");
    egl = dlopen(eglPath, RTLD_NOW | RTLD_LOCAL);
    gles = dlopen(glesPath, RTLD_NOW | RTLD_LOCAL);
    require(egl && gles, "Could not load the selected Mesa EGL/GLES libraries");
    current = symbol<decltype(current)>(egl, "eglGetCurrentContext");
#define LOAD(member, name) member = symbol<decltype(member)>(gles, name)
    LOAD(finish, "glFinish"); LOAD(error, "glGetError");
    LOAD(get, "glGetIntegerv"); LOAD(attachment, "glGetFramebufferAttachmentParameteriv");
    LOAD(bindFramebuffer, "glBindFramebuffer"); LOAD(bindBuffer, "glBindBuffer");
    LOAD(bindTexture, "glBindTexture"); LOAD(bindRenderbuffer, "glBindRenderbuffer");
    LOAD(texLevel, "glGetTexLevelParameteriv");
    LOAD(renderbuffer, "glGetRenderbufferParameteriv"); LOAD(pack, "glPixelStorei");
    LOAD(readBuffer, "glReadBuffer"); LOAD(readPixels, "glReadPixels");
    LOAD(stringIndex, "glGetStringi");
#undef LOAD
    debug.current = current; debug.error = error; debug.get = get;
#define DEBUG_LOAD(member, name) debug.member = symbol<decltype(debug.member)>(gles, name)
    DEBUG_LOAD(getPointer, "glGetPointerv");
    DEBUG_LOAD(isEnabled, "glIsEnabled");
    DEBUG_LOAD(enable, "glEnable"); DEBUG_LOAD(disable, "glDisable");
    DEBUG_LOAD(callback, "glDebugMessageCallback");
    DEBUG_LOAD(control, "glDebugMessageControl");
    DEBUG_LOAD(pushGroup, "glPushDebugGroup"); DEBUG_LOAD(popGroup, "glPopDebugGroup");
#undef DEBUG_LOAD
  }
  ~GL() { if(gles) dlclose(gles); if(egl) dlclose(egl); }
  int integer(unsigned name) const { int value = 0; get(name, &value); return value; }
  int attached(unsigned slot, unsigned name) const
  { int value = 0; attachment(0x8CA9, slot, name, &value); return value; }
  void check(const char *phase) const
  {
    unsigned code = error();
    require(code == 0, std::string(phase) + ": GL error " + std::to_string(code));
  }
};

struct Output
{
  ResourceId resource;
  TextureDescription texture;
  unsigned slot = 0, attachment = 0, mip = 0, layer = 0, width = 0, height = 0;
  int objectType = 0, objectName = 0;
  int framebuffer = 0;
  bool color = false;
};

static Output finalOutputMetadata(IReplayController *renderer)
{
  const GLPipe::State *state = renderer->GetGLPipelineState();
  require(state != nullptr, "Only OpenGL capture replay is supported");
  Output output;
  const auto &fbo = state->framebuffer.drawFBO;
  // The first enabled draw-buffer slot is deterministic. Never select a larger
  // unrelated image or infer an attachment from its workload/resource name.
  for(size_t slot = 0; slot < fbo.drawBuffers.size(); ++slot)
  {
    int index = fbo.drawBuffers[slot];
    if(index < 0) continue;
    require(size_t(index) < fbo.colorAttachments.size(), "Invalid final draw-buffer index");
    const Descriptor &view = fbo.colorAttachments[size_t(index)];
    if(view.resource == ResourceId()) continue;
    output.resource = view.resource;
    output.slot = unsigned(slot);
    output.mip = view.firstMip;
    output.layer = view.firstSlice;
    bool found = false;
    for(const auto &texture : renderer->GetTextures())
      if(texture.resourceId == output.resource) { output.texture = texture; found = true; break; }
    require(found, "Final bound attachment has no texture description");
    require(output.mip < output.texture.mips && output.mip < 32,
            "Final attachment mip is outside its resource");
    require(output.texture.msSamp == 1, "Final multisample attachment readback is unsupported");
    require(output.layer < std::max(output.texture.arraysize,
                                   std::max(1U, output.texture.depth >> output.mip)),
            "Final attachment layer is outside its resource");
    output.width = std::max(1U, output.texture.width >> output.mip);
    output.height = std::max(1U, output.texture.height >> output.mip);
    output.color = true;
    break;
  }
  return output;
}

static void verifyLiveOutput(GL &gl, Output &output)
{
  require(gl.current() != nullptr, "Replay returned without a current Mesa EGL context");
  const int fbo = gl.integer(0x8CA6); // GL_DRAW_FRAMEBUFFER_BINDING
  if(output.framebuffer)
    require(output.framebuffer == fbo, "Final replay framebuffer differs from capture end-state");
  output.framebuffer = fbo;
  if(!output.color)
  {
    const int count = gl.integer(0x8824); // GL_MAX_DRAW_BUFFERS
    for(int i = 0; i < count; ++i)
    {
      unsigned slot = unsigned(gl.integer(0x8825U + unsigned(i)));
      if(slot && (!fbo || gl.attached(slot, 0x8CD0) != 0))
        throw std::runtime_error("Metadata says no color, but live FBO has a color attachment");
    }
    gl.check("No-color framebuffer verification");
    return;
  }
  require(fbo != 0, "Physical default framebuffer extent/resource mapping is unsupported");
  output.attachment = unsigned(gl.integer(0x8825U + output.slot));
  require(output.attachment != 0, "Final output slot is not live in the completed FBO");
  const int objectType = gl.attached(output.attachment, 0x8CD0);
  const int objectName = gl.attached(output.attachment, 0x8CD1);
  if(output.objectName)
    require(output.objectType == objectType && output.objectName == objectName,
            "Final replay attachment identity differs from capture end-state");
  output.objectType = objectType;
  output.objectName = objectName;
  const int component = gl.attached(output.attachment, 0x8211);
  require(component != 0x1404 && component != 0x1405,
          "Integer final attachment PNG conversion is unsupported");
  int width = 0, height = 0;
  if(output.objectType == 0x1702) // GL_TEXTURE
  {
    const int mip = gl.attached(output.attachment, 0x8CD2);
    int layer = gl.attached(output.attachment, 0x8CD4);
    const int face = gl.attached(output.attachment, 0x8CD3);
    require(gl.attached(output.attachment, 0x8DA7) == 0,
            "Layered final attachment is unsupported");
    if(face >= 0x8515 && face <= 0x851A) layer = face - 0x8515;
    require(mip == int(output.mip) && layer == int(output.layer),
            "Live attachment subresource differs from final capture pipeline state");
    unsigned target = 0, binding = 0, levelTarget = 0;
    switch(output.texture.type)
    {
      case TextureType::Texture2D: target = 0x0DE1; binding = 0x8069; break;
      case TextureType::Texture2DArray: target = 0x8C1A; binding = 0x8C1D; break;
      case TextureType::Texture3D: target = 0x806F; binding = 0x806A; break;
      case TextureType::TextureCube: target = 0x8513; binding = 0x8514; levelTarget = unsigned(face); break;
      default: throw std::runtime_error("Final attachment texture target is unsupported");
    }
    if(!levelTarget) levelTarget = target;
    const int previous = gl.integer(binding);
    gl.bindTexture(target, unsigned(output.objectName));
    gl.texLevel(levelTarget, mip, 0x1000, &width);
    gl.texLevel(levelTarget, mip, 0x1001, &height);
    gl.bindTexture(target, unsigned(previous));
  }
  else if(output.objectType == 0x8D41) // GL_RENDERBUFFER
  {
    require(output.mip == 0 && output.layer == 0, "Renderbuffer has nonzero subresource");
    const int previous = gl.integer(0x8CA7);
    gl.bindRenderbuffer(0x8D41, unsigned(output.objectName));
    gl.renderbuffer(0x8D41, 0x8D42, &width);
    gl.renderbuffer(0x8D41, 0x8D43, &height);
    gl.bindRenderbuffer(0x8D41, unsigned(previous));
  }
  else throw std::runtime_error("Final attachment has no live texture/renderbuffer");
  gl.check("Live final attachment verification");
  require(width == int(output.width) && height == int(output.height),
          "Live attachment extent differs from final capture pipeline state");
}

static std::vector<uint8_t> readCompletedColor(GL &gl, const Output &output)
{
  require(output.width <= unsigned(std::numeric_limits<int>::max()) &&
          output.height <= unsigned(std::numeric_limits<int>::max()) &&
          output.width <= std::numeric_limits<size_t>::max() / 4 / output.height,
          "Final attachment readback dimensions overflow");
  std::vector<uint8_t> pixels(size_t(output.width) * output.height * 4);
  const int drawFbo = gl.integer(0x8CA6), readFbo = gl.integer(0x8CAA);
  const int oldReadBuffer = gl.integer(0x0C02), pbo = gl.integer(0x88ED);
  const std::array<unsigned, 4> parameters{{0x0D05, 0x0D02, 0x0D03, 0x0D04}};
  std::array<int, 4> saved{};
  for(size_t i = 0; i < saved.size(); ++i) saved[i] = gl.integer(parameters[i]);
  bool reverseAvailable = false;
  const int extensions = gl.integer(0x821D);
  for(int i = 0; i < extensions; ++i)
  {
    const auto *extension = gl.stringIndex(0x1F03, unsigned(i));
    if(extension && std::string(reinterpret_cast<const char *>(extension)) == "GL_ANGLE_pack_reverse_row_order")
      reverseAvailable = true;
  }
  int reverse = reverseAvailable ? gl.integer(0x93A4) : 0;
  gl.check("Readback state capture");
  gl.bindFramebuffer(0x8CA8, unsigned(drawFbo));
  const int drawFboReadBuffer = gl.integer(0x0C02);
  gl.readBuffer(output.attachment);
  gl.bindBuffer(0x88EB, 0);
  for(size_t i = 0; i < saved.size(); ++i) gl.pack(parameters[i], i == 0 ? 1 : 0);
  if(reverseAvailable) gl.pack(0x93A4, 0);
  gl.readPixels(0, 0, int(output.width), int(output.height), 0x1908, 0x1401, pixels.data());
  const unsigned readError = gl.error();
  for(size_t i = 0; i < saved.size(); ++i) gl.pack(parameters[i], saved[i]);
  if(reverseAvailable) gl.pack(0x93A4, reverse);
  gl.bindBuffer(0x88EB, unsigned(pbo));
  gl.readBuffer(unsigned(drawFboReadBuffer));
  gl.bindFramebuffer(0x8CA8, unsigned(readFbo));
  gl.readBuffer(unsigned(oldReadBuffer));
  gl.check("Readback state restoration");
  require(readError == 0, "Completed attachment glReadPixels failed: " + std::to_string(readError));
  return pixels;
}

static void countActions(const rdcarray<ActionDescription> &actions, uint32_t &end, size_t &draws)
{
  for(const auto &action : actions)
  {
    end = std::max(end, action.eventId);
    if((action.flags & ActionFlags::Drawcall) != ActionFlags::NoFlags) ++draws;
    countActions(action.children, end, draws);
  }
}

static bool typeFlag(const SDObject &object, SDTypeFlags flag)
{ return (object.type.flags & flag) != SDTypeFlags::NoFlags; }

static void validateDebugObservation(const SDFile &file)
{
  uint64_t groups = 0;
  for(const SDChunk *chunk : file.chunks)
  {
    if(!chunk) continue;
    bool hasCap = false;
    uint64_t cap = 0;
    for(size_t i = 0; i < chunk->NumChildren(); ++i)
    {
      const SDObject *child = chunk->GetChild(i);
      if(child && child->name == "cap" &&
         (child->type.basetype == SDBasic::Enum || child->type.basetype == SDBasic::UnsignedInteger))
      { hasCap = true; cap = child->data.basic.u; }
    }
    pvrgpu::rdc::ValidateDebugCaptureCall(chunk->name.c_str(), hasCap, cap, groups);
  }
}

static void traceValue(std::ostream &out, const SDObject &object)
{
  if(typeFlag(object, SDTypeFlags::NullString)) { out << "NULL"; return; }
  switch(object.type.basetype)
  {
    case SDBasic::Chunk:
    case SDBasic::Struct:
    case SDBasic::Array:
    {
      const bool array = object.type.basetype == SDBasic::Array;
      out << (array ? '[' : '{');
      bool first = true;
      for(size_t i = 0; i < object.NumChildren(); ++i)
      {
        const SDObject *child = object.GetChild(i);
        if(!child || typeFlag(*child, SDTypeFlags::Hidden)) continue;
        if(!first) out << ", ";
        first = false;
        if(!array) out << child->name.c_str() << '=';
        traceValue(out, *child);
      }
      out << (array ? ']' : '}');
      break;
    }
    case SDBasic::Null: out << "NULL"; break;
    case SDBasic::Buffer: out << "<buffer " << object.type.byteSize << " bytes>"; break;
    case SDBasic::String: out << json(object.data.str.c_str()); break;
    case SDBasic::Enum:
    case SDBasic::UnsignedInteger:
      if(typeFlag(object, SDTypeFlags::HasCustomString) && !object.data.str.empty())
        out << object.data.str.c_str();
      else out << object.data.basic.u;
      break;
    case SDBasic::SignedInteger: out << object.data.basic.i; break;
    case SDBasic::Float: out << std::setprecision(17) << object.data.basic.d; break;
    case SDBasic::Boolean: out << (object.data.basic.b ? "true" : "false"); break;
    case SDBasic::Character: out << json(std::string(1, object.data.basic.c)); break;
    case SDBasic::Resource: out << "ResourceId(" << object.data.basic.u << ')'; break;
    case SDBasic::GPUAddress: out << "0x" << std::hex << object.data.basic.u << std::dec; break;
  }
}

static void writeTrace(const SDFile &file, const fs::path &path, const fs::path &source)
{
  std::ofstream out(path);
  require(bool(out), "Could not create API trace");
  out << "# API Trace\n\nSource: " << source.string() << "\n\n```text\n";
  for(const SDChunk *chunk : file.chunks)
  {
    if(!chunk) continue;
    const std::string name = chunk->name.c_str();
    if(name.compare(0, 10, "Internal::") == 0 || name == "DeviceInitialisation") continue;
    out << name << " (";
    bool first = true;
    for(size_t i = 0; i < chunk->NumChildren(); ++i)
    {
      const SDObject *child = chunk->GetChild(i);
      if(!child || typeFlag(*child, SDTypeFlags::Hidden)) continue;
      if(!first) out << ", ";
      first = false;
      out << child->name.c_str() << '=';
      traceValue(out, *child);
    }
    out << ")\n";
  }
  out << "```\n";
  out.close();
  require(out.good(), "Could not finish API trace");
}

int main(int argc, char **argv)
{
  if(argc != 3 && argc != 4)
  {
    std::cerr << "Usage: " << argv[0] << " TRACE.rdc OUTPUT.png [OUTPUT_trace.md]\n";
    return 2;
  }
  fs::path receipt;
  try
  {
    const fs::path capture = fs::canonical(argv[1]);
    // Resolve existing parent symlinks too: different spellings must not let
    // optional trace output overwrite a capture, PNG, or receipt.
    const fs::path png = fs::weakly_canonical(fs::absolute(argv[2]));
    const fs::path trace = argc == 4 ? fs::weakly_canonical(fs::absolute(argv[3])) : fs::path{};
    if(const char *path = std::getenv("PVRGPU_RDC_FINAL_OUTPUT_RECEIPT"))
      if(*path) receipt = fs::weakly_canonical(fs::absolute(path));
    require(capture != png && capture != receipt && (receipt.empty() || png != receipt) &&
            (trace.empty() || (trace != capture && trace != png && trace != receipt)),
            "Input/output paths must be distinct");
    require(!fs::exists(png) && (receipt.empty() || !fs::exists(receipt)) &&
            (trace.empty() || !fs::exists(trace)),
            "Refusing existing final PNG/receipt; use a fresh output path");
    const std::string backend = std::getenv("GALLIUM_DRIVER") ? std::getenv("GALLIUM_DRIVER") : "unknown";
    require(backend != "pvrgpu" || !receipt.empty(), "PvrGPU replay requires a final-output receipt path");
    if(backend == "pvrgpu")
    {
      const char *api = std::getenv("PVRGPU_SYSTEMC_API_LIB");
      const char *counter = std::getenv("PVRGPU_DRIVER_COUNTER_OUT");
      const char *report = std::getenv("PVRGPU_SYSTEMC_JSONL_OUT");
      require(api && *api && fs::is_regular_file(api) && counter && *counter && report && *report,
              "PvrGPU formal replay requires native API library, driver counter and model JSONL paths");
    }
    const fs::path initialAudit = backend == "pvrgpu" ?
        fs::path(receipt.string() + ".initial-copy-driver-counter.txt") : fs::path{};
    require(initialAudit.empty() || (initialAudit != capture && initialAudit != png && initialAudit != trace &&
            initialAudit != fs::weakly_canonical(fs::absolute(std::getenv("PVRGPU_DRIVER_COUNTER_OUT")))),
            "Initial-copy audit paths must be distinct");
    require(initialAudit.empty() || !fs::exists(initialAudit), "Refusing existing initial-copy driver audit");
    NativeScope native;
    // The action count is reporting metadata, never permission to finish work.
    unsetenv("PVRGPU_RDC_TRACE_DRAW_ACTIONS");
    unsetenv("PVRGPU_RDC_CASE_NAME");
    ReplayLifetime replay;
    struct DisableBeforeShutdown { NativeScope &scope; ~DisableBeforeShutdown() { scope.disable(); } } disable{native};
    replay.file = RENDERDOC_OpenCaptureFile();
    require(replay.file != nullptr, "Could not create RenderDoc capture reader");
    auto opened = replay.file->OpenFile(capture.string().c_str(), "rdc", nullptr);
    require(opened.code == ResultCode::Succeeded, "OpenFile failed: " + std::to_string(unsigned(opened.code)));
    for(int i = 0; i < replay.file->GetSectionCount(); ++i)
    {
      const std::string name = replay.file->GetSectionProperties(i).name.c_str();
      for(const std::string suffix : {"/rdc-slice", "/rdc-split"})
        require(name.size() < suffix.size() || name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0,
                "Slice replay requires an explicit dependency contract; formal player supports complete frames only");
    }
    ReplayOptions options;
    options.apiValidation = true;
    auto openedReplay = replay.file->OpenCapture(options, nullptr);
    const auto &status = openedReplay.first;
    replay.renderer = openedReplay.second;
    require(status.code == ResultCode::Succeeded && replay.renderer,
            "OpenCapture failed: " + std::to_string(unsigned(status.code)));
    uint32_t endEvent = 0;
    size_t draws = 0;
    countActions(replay.renderer->GetRootActions(), endEvent, draws);
    require(endEvent != 0, "Capture has no replayable action boundary");
    validateDebugObservation(replay.renderer->GetStructuredFile());
    // Only metadata is retained from this native-isolated preview. No golden
    // pixels are read or copied. Initial contents are restored again below.
    replay.renderer->SetFrameEvent(endEvent, true);
    Output output = finalOutputMetadata(replay.renderer);
    GL gl;
    for(unsigned i = 0; i < 64 && gl.error(); ++i) {}
    gl.check("Initialization error reset before attachment metadata");
    verifyLiveOutput(gl, output);
    if(!trace.empty()) writeTrace(replay.renderer->GetStructuredFile(), trace, capture);
    // Preview work is isolated and is not the required initial resource copy.
    // Clear only its old diagnostics, then retain every error from restoration
    // itself, even when RenderDoc internally drains the sticky GL error flag.
    for(unsigned i = 0; i < 64 && gl.error(); ++i) {}
    gl.check("Preview error reset before initial resource restoration");
    replay.renderer->GetDebugMessages();
    if(!initialAudit.empty())
    {
      fs::create_directories(initialAudit.parent_path());
      const int fd = open(initialAudit.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
      require(fd >= 0, "Could not exclusively create initial-copy driver audit");
      require(close(fd) == 0, "Could not close initial-copy driver audit");
      require(setenv("PVRGPU_DRIVER_COUNTER_OUT", initialAudit.c_str(), 1) == 0,
              "Could not enable isolated initial-copy audit");
    }
    {
      pvrgpu::rdc::GLDebugScope initialErrors(gl.debug);
      replay.renderer->SetFrameEvent(0, true);
      initialErrors.Check("Initial resource restoration");
      gl.finish();
      initialErrors.Finish();
    }
    require(replay.renderer->GetFatalErrorStatus().code == ResultCode::Succeeded,
            "RenderDoc failed to restore initial capture state");
    require(gl.current() != nullptr, "Initial-state restoration left no replay EGL context");
    void *replayContext = gl.current();
    for(const auto &message : replay.renderer->GetDebugMessages())
      require(message.severity != MessageSeverity::High &&
              message.source != MessageSource::IncorrectAPIUse &&
              message.source != MessageSource::UnsupportedConfiguration,
              "Initial resource restoration validation error: " + std::string(message.description.c_str()));
    if(!initialAudit.empty())
    {
      unsetenv("PVRGPU_DRIVER_COUNTER_OUT");
      std::ifstream audit(initialAudit);
      require(bool(audit), "Could not read initial-copy driver audit");
      const std::string text{std::istreambuf_iterator<char>(audit), {}};
      require(!audit.bad(), "Could not finish reading initial-copy driver audit");
      std::string error;
      require(pvrgpu::rdc::ValidateInitialCopyAudit(text, &error), error);
    }
    pvrgpu::rdc::GLDebugScope apiErrors(gl.debug);
    native.enable();
    require(replay.renderer->ReplayEventRange(1, endEvent), "Complete ordered replay failed");
    require(gl.current() == replayContext, "Replay returned in a different EGL context");
    gl.check("Formal replay");
    gl.finish();
    gl.check("Formal replay glFinish");
    apiErrors.Check("Formal replay and glFinish");
    require(replay.renderer->GetFatalErrorStatus().code == ResultCode::Succeeded,
            "RenderDoc reported fatal replay failure");
    for(const auto &message : replay.renderer->GetDebugMessages())
      require(message.severity != MessageSeverity::High &&
              message.source != MessageSource::IncorrectAPIUse &&
              message.source != MessageSource::UnsupportedConfiguration,
              "Replay validation error at event " + std::to_string(message.eventId) +
              ": " + message.description.c_str());
    verifyLiveOutput(gl, output);
    std::vector<uint8_t> pixels;
    if(output.color) pixels = readCompletedColor(gl, output);
    apiErrors.Finish();
    // No GetTextureData/SaveTexture debug draw can enter the native scope.
    native.disable();
    if(output.color) pvrgpu::stub::WriteRgbaPngAtomic(png, pixels, output.width, output.height);
    if(!receipt.empty())
    {
      fs::create_directories(receipt.parent_path());
      const fs::path part = receipt.string() + ".part";
      std::ofstream out(part);
      require(bool(out), "Could not write final-output receipt");
      out << "{\"schema\":\"pvrgpu.rdc-final-output.v2\",\"backend\":" << json(backend)
          << ",\"status\":\"PASS\",\"rdc_path\":" << json(capture.string())
          << ",\"replay_begin_event\":1,\"replay_end_event\":" << endEvent
          << ",\"trace_draw_actions\":" << draws
          << ",\"initial_native_isolated\":true,\"replay_completed\":true"
          << ",\"initial_contents_restored\":true"
          << ",\"replay_context_finished\":true,\"api_errors\":0,\"color_output\":"
          << (output.color ? "true" : "false")
          << ",\"api_error_capture\":\"synchronous-gl-debug-callback\",\"debug_callback_verified\":true"
          << ",\"source\":\"completed-replay-attachment\",\"readback_api\":\"replay-glReadPixels\"";
      if(!initialAudit.empty()) out << ",\"initial_copy_driver_counter_path\":" << json(initialAudit.string());
      if(output.color)
        out << ",\"resource_id\":" << json(resourceString(output.resource))
            << ",\"mip\":" << output.mip << ",\"layer\":" << output.layer
            << ",\"sample\":0,\"width\":" << output.width << ",\"height\":" << output.height
            << ",\"format\":" << json(output.texture.format.Name().c_str())
            << ",\"png_path\":" << json(png.string()) << ",\"draw_buffer_slot\":" << output.slot
            << ",\"gl_framebuffer\":" << output.framebuffer
            << ",\"gl_attachment\":" << output.attachment << ",\"gl_object_name\":" << output.objectName;
      out << "}\n";
      out.close();
      require(out.good(), "Could not finish final-output receipt");
      fs::rename(part, receipt);
    }
    std::cout << "Color output: " << (output.color ? std::to_string(output.width) + "x" + std::to_string(output.height) : "none")
              << "\nColor format: " << (output.color ? output.texture.format.Name().c_str() : "none")
              << "\nTrace draw actions: " << draws << "\nPNG: " << (output.color ? png.string() : "none") << '\n';
    return 0;
  }
  catch(const std::exception &error)
  {
    std::cerr << "Formal replay failed: " << error.what() << '\n';
    return 1;
  }
}
